"""Train and evaluate a modern agent (agents/modern.py) on one game, many games in parallel.

    uv run python -m train.modern --game aliens --preset dqn --ticks 1014134 --seed 0 --name aliens-dqn-s0

Budget is in game ticks (2018's first 2000 Aliens episodes used 1,014,134). Writes
runs/<name>/episodes.csv and results/<name>.json with the same evaluation as the 2018 runs
(50 games, greedy, levels 0-2) plus 50 games on held-out levels 3-4.
"""

import argparse
import csv
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, replace

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
    "dqn-outcome-score01-ar2-impala": lambda: ModernConfig(outcome_reward=True, score_scale=0.1, action_repeat=2, encoder="impala"),
    "dqn-outcome-score01-ar1-impala": lambda: ModernConfig(outcome_reward=True, score_scale=0.1, action_repeat=1, encoder="impala"),
}


def random_play_score(game: str) -> float:
    """Mean score of random play on the game, from results/baseline-<game>-random.json."""
    ev = json.loads((ROOT / "results" / f"baseline-{game}-random.json").read_text())["eval"]
    return float(ev["mean_score"] if "mean_score" in ev else ev["score"])


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
    env = vector(a.game, a.envs, a.seed, train_levels, cfg.action_repeat)
    obs, _ = env.reset(seed=a.seed)
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
    with open(run / "episodes.csv", "w", newline="") as f:
        out = csv.writer(f)
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
        "stopped_early_at": stopped_early,
        "commit": commit,
    }  # fmt: skip
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{a.name}.json").write_text(json.dumps(result, indent=1))
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
