"""What the trainers share: frame stacks on the GPU, finished-game records, and the greedy evaluation
every agent is scored with (50 games, a fixed seed, the game's levels)."""

import numpy as np
import torch

from agents.modern import STACK, upload
from gvgai.vec import vector


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


def finished(info: dict, ended: np.ndarray) -> list[dict]:
    fi = info.get("final_info", {})
    return [{k: fi[k][e] for k in ("score", "win", "level")} for e in np.flatnonzero(ended)]


def evaluate(agent, game: str, n_envs: int, games: int, levels, seed: int, repeat: int) -> dict:
    """Greedy play; `agent` needs .q (a module), .device and act(stacks, eps)."""
    env = vector(game, n_envs, seed, levels, repeat)
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
