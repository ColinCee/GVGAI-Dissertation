"""Train and evaluate a modern agent (agents/modern.py) on one game, many games in parallel.

    uv run python train_modern.py --game aliens --preset dqn --ticks 1014134 --seed 0 --name aliens-dqn-s0

Budget is in game ticks (2018's first 2000 Aliens episodes used 1,014,134). Writes
runs/<name>/episodes.csv and results/<name>.json with the same evaluation as the 2018 runs
(50 games, greedy, levels 0-2) plus 50 games on held-out levels 3-4.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from agents.modern import STACK, ModernAgent, ModernConfig, rainbow, upload
from gvgai import heartbeat
from gvgai.vec import vector
from train2018 import DEFAULT_LEVELS

HERE = Path(__file__).resolve().parent
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
}


class Stacks:
    """Last STACK frames per game, kept on the agent's device; a new episode starts with its first
    frame repeated. `newest` is the latest frame as numpy, for the replay."""

    def __init__(self, first: np.ndarray, device: str):
        self.device = device
        f = upload(first, device)
        self.s = f[:, None].repeat(1, STACK, 1, 1)
        self.newest = first

    def push(self, frames: np.ndarray, reset: np.ndarray) -> None:
        f = upload(frames, self.device)
        self.s = torch.cat([self.s[:, 1:], f[:, None]], 1)
        if reset.any():
            r = torch.from_numpy(reset).to(self.device)
            self.s[r] = f[r][:, None].repeat(1, STACK, 1, 1)
        self.newest = frames

    def array(self) -> torch.Tensor:
        return self.s


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def finished(info: dict, ended: np.ndarray) -> list[dict]:
    fi = info.get("final_info", {})
    return [{k: fi[k][e] for k in ("score", "win", "level")} for e in np.flatnonzero(ended)]


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


def evaluate(agent: ModernAgent, game: str, n_envs: int, games: int, levels, seed: int, repeat: int,
             processes: bool = False) -> dict:  # fmt: skip
    env = vector(game, n_envs, seed, levels, repeat, processes=processes)
    obs, _ = env.reset(seed=seed)
    stacks = Stacks(obs, agent.device)
    per_env = -(-games // n_envs)
    done_count = np.zeros(n_envs, np.int64)
    rows: list[dict] = []
    agent.q.eval()
    while (done_count < per_env).any():
        a = agent.act(stacks.array(), 0.0)
        obs, _, term, trunc, info = env.step(a)
        ended = term | trunc
        for e, r in zip(np.flatnonzero(ended), finished(info, ended), strict=True):
            if done_count[e] < per_env:
                rows.append(r)
            done_count[e] += 1
        stacks.push(obs, ended)
    agent.q.train()
    env.close()
    rows = rows[:games]
    return {
        "games": len(rows), "levels": list(levels),
        "mean_score": float(np.mean([r["score"] for r in rows])),
        "win_rate": float(np.mean([r["win"] for r in rows])),
        "scores": [float(r["score"]) for r in rows], "wins": [int(r["win"]) for r in rows],
    }  # fmt: skip


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
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--processes", action="store_true", help="old env layer: worker processes + PIL Lanczos")
    p.add_argument("--early-stop", action="store_true", help="long budget: stop once training stops improving")
    a = p.parse_args()

    commit = git_commit()  # at start: a commit made while the run trains isn't the code it ran
    cfg = PRESETS[a.preset]()
    if a.buffer:
        cfg.buffer_decisions = a.buffer
    if a.learning_starts is not None:
        cfg.learning_starts = a.learning_starts
    run = HERE / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    levels = DEFAULT_LEVELS.get(a.game, (0, 1, 2))
    env = vector(a.game, a.envs, a.seed, levels, cfg.action_repeat, processes=a.processes)
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
    ev = evaluate(agent, a.game, a.envs, a.eval_games, levels, a.seed + 7, cfg.action_repeat, a.processes)
    ev_held = evaluate(agent, a.game, a.envs, a.eval_games, (3, 4), a.seed + 8, cfg.action_repeat, a.processes)
    result = {
        "name": a.name, "agent": a.preset, "game": a.game, "seed": a.seed, "train_ticks": ticks,
        "decisions": decisions, "train_wall_s": train_wall, "env_s": t_env, "learn_s": t_learn,
        "updates": agent.updates, "envs": a.envs,
        "env_layer": "processes+PIL" if a.processes else "in-process+gray_half",
        "device": a.device, "config": asdict(cfg),
        "eval": ev, "eval_heldout": ev_held, "budget_ticks": a.ticks, "stopped_early_at": stopped_early,
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
