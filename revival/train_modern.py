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
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from agents.modern import STACK, ModernAgent, ModernConfig, rainbow
from gvgai.vec import vector

HERE = Path(__file__).resolve().parent
PRESETS = {
    "dqn": ModernConfig,  # DQN with today's defaults: Double, Huber, Adam, big replay, /255 inputs
    "dqn-per": lambda: ModernConfig(per=True),
    "dqn-per-dueling": lambda: ModernConfig(per=True, dueling=True),
    "dqn-per-dueling-nstep": lambda: ModernConfig(per=True, dueling=True, n_step=3),
    "rainbow": rainbow,
}


class Stacks:
    """Last STACK frames per game; a new episode starts with its first frame repeated."""

    def __init__(self, first: np.ndarray):
        self.q = [deque([f] * STACK, maxlen=STACK) for f in first]

    def push(self, frames: np.ndarray, reset: np.ndarray) -> None:
        for e, f in enumerate(frames):
            if reset[e]:
                self.q[e] = deque([f] * STACK, maxlen=STACK)
            else:
                self.q[e].append(f)

    def array(self) -> np.ndarray:
        return np.stack([np.stack(q) for q in self.q])


def git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def finished(info: dict, ended: np.ndarray) -> list[dict]:
    fi = info.get("final_info", {})
    return [{k: fi[k][e] for k in ("score", "win", "level")} for e in np.flatnonzero(ended)]


def evaluate(agent: ModernAgent, game: str, n_envs: int, games: int, levels, seed: int, repeat: int) -> dict:
    env = vector(game, n_envs, seed, levels, repeat)
    obs, _ = env.reset(seed=seed)
    stacks = Stacks(obs)
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
    p.add_argument("--envs", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--buffer", type=int, default=None, help="replay size in decisions")
    p.add_argument("--learning-starts", type=int, default=None)
    p.add_argument("--eval-games", type=int, default=50)
    a = p.parse_args()

    cfg = PRESETS[a.preset]()
    if a.buffer:
        cfg.buffer_decisions = a.buffer
    if a.learning_starts is not None:
        cfg.learning_starts = a.learning_starts
    run = HERE / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    env = vector(a.game, a.envs, a.seed, (0, 1, 2), cfg.action_repeat)
    obs, _ = env.reset(seed=a.seed)
    n_actions = int(env.single_action_space.n)
    h, w = obs.shape[1:]
    agent = ModernAgent(a.envs, h, w, n_actions, cfg, a.device, a.seed)
    stacks = Stacks(obs)
    budget_decisions = a.ticks // cfg.action_repeat

    ticks = decisions = episodes = 0
    pending_updates = 0.0
    t0 = time.perf_counter()
    t_env = t_learn = 0.0
    with open(run / "episodes.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["episode", "score", "win", "level", "cum_ticks", "updates", "epsilon", "wall_s"])
        while ticks < a.ticks:
            eps = agent.epsilon(decisions, budget_decisions)
            current = stacks.array()
            act = agent.act(current, eps)
            te = time.perf_counter()
            obs, r, term, trunc, info = env.step(act)
            t_env += time.perf_counter() - te
            ended = term | trunc
            final = np.array(obs)
            if ended.any():
                for e in np.flatnonzero(ended):
                    final[e] = info["final_obs"][e]
            agent.replay.add(current[:, -1], act, r, term, trunc, final)
            stacks.push(obs, ended)
            ticks += int(np.sum(info["ticks"]))
            decisions += a.envs
            for row in finished(info, ended):
                episodes += 1
                out.writerow([episodes, row["score"], int(row["win"]), row["level"], ticks, agent.updates,
                              f"{eps:.3f}", f"{time.perf_counter() - t0:.1f}"])  # fmt: skip
            if decisions >= cfg.learning_starts:
                pending_updates += a.envs * cfg.replay_ratio
                tl = time.perf_counter()
                while pending_updates >= 1:
                    agent.update(min(1.0, decisions / budget_decisions))
                    pending_updates -= 1
                t_learn += time.perf_counter() - tl
            if decisions % (a.envs * 500) == 0:
                f.flush()
                el = time.perf_counter() - t0
                print(f"ticks {ticks} ({ticks / el:.0f}/s) eps {eps:.3f} episodes {episodes} updates {agent.updates} "
                      f"env {t_env / el:.0%} learn {t_learn / el:.0%}", flush=True)  # fmt: skip
    env.close()
    train_wall = time.perf_counter() - t0
    torch.save(agent.q.state_dict(), run / "q.pt")
    ev = evaluate(agent, a.game, a.envs, a.eval_games, (0, 1, 2), a.seed + 7, cfg.action_repeat)
    ev_held = evaluate(agent, a.game, a.envs, a.eval_games, (3, 4), a.seed + 8, cfg.action_repeat)
    result = {
        "name": a.name, "agent": a.preset, "game": a.game, "seed": a.seed, "train_ticks": ticks,
        "decisions": decisions, "train_wall_s": train_wall, "env_s": t_env, "learn_s": t_learn,
        "updates": agent.updates, "envs": a.envs, "device": a.device, "config": asdict(cfg),
        "eval": ev, "eval_heldout": ev_held,
        "commit": git_commit(),
    }  # fmt: skip
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{a.name}.json").write_text(json.dumps(result, indent=1))
    print(f"done: {ticks} ticks in {train_wall / 60:.1f} min; eval {ev['mean_score']:.1f} ({ev['win_rate']:.0%}), "
          f"held-out {ev_held['mean_score']:.1f} ({ev_held['win_rate']:.0%})")  # fmt: skip


if __name__ == "__main__":
    main()
    # Interpreter teardown with spawned env workers and torch loaded sometimes segfaults on Windows,
    # after everything is written; exit directly so the exit code reflects the run.
    os._exit(0)
