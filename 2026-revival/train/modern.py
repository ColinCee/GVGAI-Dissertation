"""Train and evaluate a modern agent (agents/modern.py) on one game, many games in parallel.

    uv run python -m train.modern --game aliens --preset dqn --ticks 1014134 --seed 0 --name aliens-dqn-s0

Budget is in game ticks (2018's first 2000 Aliens episodes used 1,014,134). Writes
runs/<name>/episodes.csv and results/<name>.json with the same evaluation as the 2018 runs
(50 games, greedy, levels 0-2) plus 50 games on held-out levels 3-4.

Saves runs/<name>/checkpoint every --checkpoint-minutes; started again with the same arguments, a
stopped run carries on from it. The games in play when it stopped are lost: their last move is
dropped from replay and they restart, so a resumed run is not bit-identical to an unbroken one.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from agents.evaluate import Stacks, evaluate, finished
from agents.modern import ModernAgent, ModernConfig, rainbow
from gvgai import heartbeat
from gvgai.games import HELD_OUT, ROOT, game_levels, git_commit
from gvgai.vec import vector

HERE = ROOT


PRESETS = {
    "dqn": ModernConfig,  # DQN with today's defaults: Double, Huber, Adam, big replay, /255 inputs
    "dqn-per": lambda: ModernConfig(per=True),
    "dqn-per-dueling": lambda: ModernConfig(per=True, dueling=True),
    "dqn-per-dueling-nstep": lambda: ModernConfig(per=True, dueling=True, n_step=3),
    "rainbow": rainbow,
    # reward experiments (modern DQN): the dissertation's reward, outcome only, outcome + longer horizon
    "dqn-reward2018": lambda: ModernConfig(outcome_reward=True, loss_penalty=0.5),
    "dqn-outcome": lambda: ModernConfig(outcome_reward=True),
    "dqn-outcome-g997": lambda: ModernConfig(outcome_reward=True, gamma=0.997),
    "dqn-penalty-only": lambda: ModernConfig(loss_penalty=0.5),
    "dqn-reward2018-p1": lambda: ModernConfig(outcome_reward=True, loss_penalty=1.0),
    "dqn-reward2018-raw-mse": lambda: ModernConfig(
        outcome_reward=True, loss_penalty=0.5, scale_input=False, huber=False
    ),
    "rainbow-reward2018": lambda: replace(rainbow(), outcome_reward=True, loss_penalty=0.5),
    "dqn-impala": lambda: ModernConfig(encoder="impala"),
    "rainbow-impala": lambda: replace(rainbow(), encoder="impala"),
    "dqn-reward2018-impala": lambda: ModernConfig(outcome_reward=True, loss_penalty=0.5, encoder="impala"),
    "rainbow-reward2018-impala": lambda: replace(rainbow(), outcome_reward=True, loss_penalty=0.5, encoder="impala"),
    "dqn-outcome-only": lambda: ModernConfig(outcome_reward=True, score_scale=0.0),
    "dqn-outcome-score01": lambda: ModernConfig(outcome_reward=True, score_scale=0.1),
    "dqn-outcome-only-nstep": lambda: ModernConfig(outcome_reward=True, score_scale=0.0, n_step=3),
    "dqn-outcome5": lambda: ModernConfig(outcome_reward=True, win_reward=5.0, loss_reward=-5.0),
    "dqn-outcome10": lambda: ModernConfig(outcome_reward=True, win_reward=10.0, loss_reward=-10.0),
    "dqn-loss-only": lambda: ModernConfig(outcome_reward=True, win_reward=0.0, loss_reward=-1.0),
    "dqn-penalty-decay": lambda: ModernConfig(outcome_reward=True, loss_penalty=0.5, penalty_decay=0.97),
    # loss -1; a win 0.5 to 1 by its score, against random play's score (set from the baseline below)
    "dqn-outcome-saturating": lambda: ModernConfig(outcome_reward=True, score_scale=0.0, win_score_ref=-1.0),
    # winning and scoring together (23 Oct 2026): outcome + 0.1 x score, varied one thing at a time
    "dqn-outcome-score01-n10": lambda: ModernConfig(outcome_reward=True, score_scale=0.1, n_step=10),
    "dqn-outcome-score01-n20": lambda: ModernConfig(outcome_reward=True, score_scale=0.1, n_step=20),
    "dqn-outcome-score01-ar2": lambda: ModernConfig(outcome_reward=True, score_scale=0.1, action_repeat=2),
    "dqn-outcome-score01-impala": lambda: ModernConfig(outcome_reward=True, score_scale=0.1, encoder="impala"),
    "dqn-outcome-score005": lambda: ModernConfig(outcome_reward=True, score_scale=0.05),
    "dqn-outcome-score02": lambda: ModernConfig(outcome_reward=True, score_scale=0.2),
    # follow-ups to a move every 2 frames scoring 76: every frame, a lighter score weight, and the IMPALA network
    "dqn-outcome-score01-ar1": lambda: ModernConfig(outcome_reward=True, score_scale=0.1, action_repeat=1),
    "dqn-outcome-score005-ar2": lambda: ModernConfig(outcome_reward=True, score_scale=0.05, action_repeat=2),
    "dqn-outcome-score01-ar2-impala": lambda: ModernConfig(
        outcome_reward=True, score_scale=0.1, action_repeat=2, encoder="impala"
    ),
    "dqn-outcome-score01-ar1-impala": lambda: ModernConfig(
        outcome_reward=True, score_scale=0.1, action_repeat=1, encoder="impala"
    ),
}


def random_play_score(game: str) -> float:
    """Mean score of random play on the game, from results/baseline-<game>-random.json."""
    ev = json.loads((ROOT / "results" / f"baseline-{game}-random.json").read_text())["eval"]
    return float(ev["mean_score"] if "mean_score" in ev else ev["score"])


REPLAY_FIELDS = ("action", "reward", "term", "end", "start", "weight", "max_p", "t")


def save_checkpoint(path: Path, agent: ModernAgent, counters: dict) -> None:
    """Everything a run needs to carry on, written beside the old checkpoint and swapped in at the end.
    Replay frames (GBs) go one game at a time to keep host memory small."""
    tmp = path.with_suffix(".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    rp = agent.replay
    frames = np.lib.format.open_memmap(tmp / "frames.npy", "w+", np.uint8, tuple(rp.frames.shape))
    for e in range(rp.n):
        frames[e] = rp.frames[e].cpu().numpy()
    frames.flush()
    del frames
    torch.save({
        "q": agent.q.state_dict(), "target": agent.target.state_dict(), "opt": agent.opt.state_dict(),
        "updates": agent.updates, "steps": agent.steps, "rng": agent.rng.bit_generator.state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        "replay": {k: getattr(rp, k).cpu() for k in REPLAY_FIELDS},
        "replay_host": {"t_host": rp.t_host, "ep_start": rp.ep_start, "ep_score": rp.ep_score},
        "counters": counters,
    }, tmp / "state.pt")  # fmt: skip
    shutil.rmtree(path, ignore_errors=True)
    tmp.rename(path)


def load_checkpoint(path: Path, agent: ModernAgent) -> dict:
    """Restore in place (the CUDA graph and compiled nets keep pointing at the same tensors) and
    return the loop's counters. The games restart, so each game's last stored move is closed off."""
    ck = torch.load(path / "state.pt", weights_only=False)
    agent.q.load_state_dict(ck["q"])
    agent.target.load_state_dict(ck["target"])
    agent.opt.load_state_dict(ck["opt"])
    agent.updates, agent.steps = ck["updates"], ck["steps"]
    agent.rng.bit_generator.state = ck["rng"]
    torch.set_rng_state(ck["torch_rng"])
    if ck["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ck["cuda_rng"])
    rp = agent.replay
    frames = np.load(path / "frames.npy", mmap_mode="r")
    for e in range(rp.n):
        rp.frames[e].copy_(torch.from_numpy(np.array(frames[e])))
    for k, v in ck["replay"].items():
        getattr(rp, k).copy_(v)
    for k, v in ck["replay_host"].items():
        getattr(rp, k)[:] = v
    for e in range(rp.n):  # the stopped game's last move has no next frame: end it, never sample it
        if rp.t_host[e] > 0:
            k = (int(rp.t_host[e]) - 1) % rp.cap
            rp.end[e, k] = True
            rp.weight[e * rp.cap + k] = 0
    rp.ep_start[:] = rp.t_host
    rp.ep_score[:] = 0
    for net in agent.actors:
        net.load_state_dict(agent.q.state_dict())
    return ck["counters"]


def flattened(history: list[tuple[int, float, int]], ticks: int, min_ticks: int = 3_000_000) -> bool:
    """The plan's rule for long runs: compare the last 20% of ticks with the 20% before it; stop when
    wins rose by under 2 points and mean score by under 2%. history: (cum_ticks, score, win) per game."""
    if ticks < min_ticks:
        return False
    w = 0.2 * ticks
    recent = [(sc, wn) for t, sc, wn in history if t > ticks - w]
    prior = [(sc, wn) for t, sc, wn in history if ticks - 2 * w < t <= ticks - w]
    if len(recent) < 20 or len(prior) < 20:
        return False
    win_gain = 100 * (np.mean([x[1] for x in recent]) - np.mean([x[1] for x in prior]))
    s_prior = np.mean([x[0] for x in prior])
    score_gain = (np.mean([x[0] for x in recent]) - s_prior) / max(abs(s_prior), 0.5)
    return win_gain < 2 and score_gain < 0.02


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="aliens")
    p.add_argument("--preset", choices=PRESETS, default="dqn")
    p.add_argument("--ticks", type=int, default=1_014_134)
    p.add_argument("--envs", type=int, default=8)  # 8 cores: 8 games beat 16 (17k vs 11k ticks/s)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--buffer", type=int, default=None, help="replay size in decisions")
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--replay-ratio", type=float, default=None, help="updates per decision (default 0.25)")
    p.add_argument("--train-levels", default=None, help="e.g. 0,1,2,3,4 (default: the game's usual levels)")
    p.add_argument("--early-stop", action="store_true", help="long budget: stop once training stops improving")
    p.add_argument("--checkpoint-minutes", type=float, default=20)
    a = p.parse_args()

    commit = git_commit()
    cfg = PRESETS[a.preset]()
    if cfg.win_score_ref < 0:  # the saturating win reward's reference: random play on this game
        cfg.win_score_ref = random_play_score(a.game)
    if a.buffer:
        cfg.buffer_decisions = a.buffer
    if a.replay_ratio is not None:
        cfg.replay_ratio = a.replay_ratio
    run = HERE / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    levels = game_levels(a.game)
    train_levels = tuple(int(x) for x in a.train_levels.split(",")) if a.train_levels else levels
    ckpt = run / "checkpoint"
    resumes = 0
    if ckpt.exists():
        resumes = torch.load(ckpt / "state.pt", weights_only=False)["counters"]["resumes"] + 1
    # a resumed run's games start from new seeds, not the ones the stopped run already played
    game_seed = a.seed + 7919 * resumes
    env = vector(a.game, a.envs, game_seed, train_levels, cfg.action_repeat)
    obs, _ = env.reset(seed=game_seed)
    n_actions = int(env.single_action_space.n)
    h, w = obs.shape[1:]
    agent = ModernAgent(a.envs, h, w, n_actions, cfg, a.device, a.seed)
    stacks = Stacks(obs, agent.device)
    if agent.act_stream:
        agent.act_stream.wait_stream(torch.cuda.current_stream())  # the first stack was built on the default stream
    budget_decisions = a.ticks // cfg.action_repeat

    ticks = decisions = episodes = 0
    history: list[tuple[int, float, int]] = []
    next_check = 250_000
    stopped_early = None
    flat_streak = 0
    pending_updates = 0.0
    t0 = time.perf_counter()
    t_env = t_learn = 0.0
    if resumes:
        c = load_checkpoint(ckpt, agent)
        ticks, decisions, episodes, history = c["ticks"], c["decisions"], c["episodes"], c["history"]
        next_check, flat_streak, pending_updates = c["next_check"], c["flat_streak"], c["pending_updates"]
        t_env, t_learn = c["t_env"], c["t_learn"]
        t0 -= c["wall"]
        print(f"resumed at {ticks} ticks (resume {resumes})", flush=True)
    last_ckpt = time.perf_counter()
    csv_path = run / "episodes.csv"
    if resumes:  # drop rows written after the checkpoint
        lines = csv_path.read_text().splitlines()[: episodes + 1]
        csv_path.write_text("\n".join(lines) + "\n")
    with open(csv_path, "a" if resumes else "w", newline="") as f:
        out = csv.writer(f)
        if not resumes:
            out.writerow(["episode", "score", "win", "level", "cum_ticks", "updates", "epsilon", "wall_s"])
        while ticks < a.ticks:
            eps = agent.epsilon(decisions, budget_decisions)
            if decisions % (a.envs * 100) == 0:
                heartbeat.check()
            newest = stacks.newest
            act = agent.act(stacks.array(), eps, actor=True)
            # The games play this move while the learner trains on what is already in replay.
            env.step_async(act)
            if decisions >= cfg.learning_starts:
                pending_updates += a.envs * cfg.replay_ratio
                tl = time.perf_counter()
                while pending_updates >= 1:
                    agent.update(min(1.0, decisions / budget_decisions))
                    pending_updates -= 1
                t_learn += time.perf_counter() - tl
            agent.refresh_actor()
            te = time.perf_counter()
            obs, r, term, trunc, info = env.step_wait()
            t_env += time.perf_counter() - te  # time spent waiting on the games beyond the learning
            ended = term | trunc
            final = np.array(obs)
            if ended.any():
                for e in np.flatnonzero(ended):
                    final[e] = info["final_obs"][e]
            won = info["final_info"]["win"] if ended.any() else None
            agent.replay.add(newest, act, r, term, trunc, final, won)
            with torch.cuda.stream(agent.act_stream) if agent.act_stream else nullcontext():
                stacks.push(obs, ended)  # on the acting stream, so it doesn't queue behind updates
            ticks += int(np.sum(info["ticks"]))
            decisions += a.envs
            for row in finished(info, ended):
                episodes += 1
                history.append((ticks, float(row["score"]), int(row["win"])))
                out.writerow([episodes, row["score"], int(row["win"]), row["level"], ticks, agent.updates,
                              f"{eps:.3f}", f"{time.perf_counter() - t0:.1f}"])  # fmt: skip
            if a.early_stop and ticks >= next_check:
                next_check += 250_000
                # two flat checks in a row: one alone stopped Rainbow on a mid-run plateau (Aliens, 1.75M)
                flat_streak = flat_streak + 1 if flattened(history, ticks) else 0
                if flat_streak >= 2:
                    stopped_early = ticks
                    print(f"stopping early at {ticks} ticks: no longer improving", flush=True)
                    break
            if time.perf_counter() - last_ckpt > a.checkpoint_minutes * 60:
                f.flush()
                tc = time.perf_counter()
                save_checkpoint(ckpt, agent, {
                    "ticks": ticks, "decisions": decisions, "episodes": episodes, "history": history,
                    "next_check": next_check, "flat_streak": flat_streak, "pending_updates": pending_updates,
                    "t_env": t_env, "t_learn": t_learn, "wall": tc - t0, "resumes": resumes,
                })  # fmt: skip
                last_ckpt = time.perf_counter()
                t0 += last_ckpt - tc  # saving is not training time
                print(f"checkpoint at {ticks} ticks ({last_ckpt - tc:.0f} s)", flush=True)
            if decisions % (a.envs * 500) == 0:
                f.flush()
                el = time.perf_counter() - t0
                print(f"ticks {ticks} ({ticks / el:.0f}/s) eps {eps:.3f} episodes {episodes} updates {agent.updates} "
                      f"env {t_env / el:.0%} learn {t_learn / el:.0%}", flush=True)  # fmt: skip
    env.close()
    train_wall = time.perf_counter() - t0
    torch.save(agent.q.state_dict(), run / "q.pt")
    ev = evaluate(agent, a.game, a.envs, a.eval_games, levels, a.seed + 7, cfg.action_repeat)
    ev_held = evaluate(agent, a.game, a.envs, a.eval_games, HELD_OUT, a.seed + 8, cfg.action_repeat)
    result = {
        "name": a.name, "agent": a.preset, "game": a.game, "seed": a.seed, "train_ticks": ticks,
        "decisions": decisions, "train_wall_s": train_wall, "env_s": t_env, "learn_s": t_learn,
        "updates": agent.updates, "envs": a.envs,
        "env_layer": "in-process+gray_half",
        "device": a.device, "config": asdict(cfg),
        "eval": ev, "eval_heldout": ev_held, "budget_ticks": a.ticks, "train_levels": list(train_levels),
        "stopped_early_at": stopped_early, "resumes": resumes,
        "commit": commit,
    }  # fmt: skip
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{a.name}.json").write_text(json.dumps(result, indent=1))
    shutil.rmtree(ckpt, ignore_errors=True)  # GBs of replay, no longer needed once the result is written
    print(f"done: {ticks} ticks in {train_wall / 60:.1f} min; eval {ev['mean_score']:.1f} ({ev['win_rate']:.0%}), "
          f"held-out {ev_held['mean_score']:.1f} ({ev_held['win_rate']:.0%})")  # fmt: skip


if __name__ == "__main__":
    if "--profile" in sys.argv:  # print where the main process spends its time
        sys.argv.remove("--profile")
        import cProfile
        import pstats

        prof = cProfile.Profile()
        prof.runcall(main)
        pstats.Stats(prof).sort_stats("tottime").print_stats(20)
    else:
        main()
    # Interpreter teardown with spawned env workers and torch loaded sometimes segfaults on Windows,
    # after everything is written; exit directly so the exit code reflects the run.
    os._exit(0)
