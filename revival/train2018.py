"""Train and evaluate the 2018 agent (agents/ddqn2018.py) on one game.

    uv run python train2018.py --game aliens --episodes 2000 --seed 0 --name aliens-2018-s0

Writes runs/<name>/episodes.csv (2018's reward_history columns plus timing), a checkpoint every
--checkpoint-minutes (a stopped run resumes from it), and results/<name>.json with the 2018-style
evaluation: 50 games, greedy, levels 0-2, weights from the last target sync.
"""

import argparse
import csv
import json
import os
import pickle
import platform
import random
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from agents.ddqn2018 import Agent2018, Config, play_episode
from gvgai.env import GvgaiEnv

HERE = Path(__file__).resolve().parent
COLUMNS = ["episode", "score", "total_reward", "steps", "win", "epsilon", "ticks", "trained", "warmup",
           "level", "cum_ticks", "wall_s"]  # fmt: skip


def save_checkpoint(path: Path, agent: Agent2018, episode: int, cum_ticks: int, wall: float) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    rp = agent.replay
    np.save(tmp / "state.npy", rp.state)
    torch.save(
        {
            "q": agent.q.state_dict(), "target": agent.target.state_dict(), "opt": agent.opt.state_dict(),
            "synced": agent.synced_state, "updates": agent.updates, "epsilon": agent.epsilon,
            "total_stacks": agent.total_stacks, "stacks_since_sync": agent.stacks_since_sync,
            "episode": episode, "cum_ticks": cum_ticks, "wall": wall,
            "torch_rng": torch.get_rng_state(),
        },
        tmp / "agent.pt",
    )  # fmt: skip
    with open(tmp / "rest.pkl", "wb") as f:
        pickle.dump({"tree": rp.tree, "action": rp.action, "reward": rp.reward, "done": rp.done,
                     "terminal_next": rp.terminal_next, "rng": agent.rng.getstate(),
                     "np_rng": agent.np_rng.bit_generator.state, "py_rng": random.getstate()}, f)  # fmt: skip
    if path.exists():
        for f in path.iterdir():
            f.unlink()
        path.rmdir()
    tmp.rename(path)


def load_checkpoint(path: Path, agent: Agent2018) -> tuple[int, int, float]:
    rp = agent.replay
    rp.state[:] = np.load(path / "state.npy", mmap_mode="r")
    ck = torch.load(path / "agent.pt", weights_only=False)
    agent.q.load_state_dict(ck["q"])
    agent.target.load_state_dict(ck["target"])
    agent.opt.load_state_dict(ck["opt"])
    agent.synced_state, agent.updates, agent.epsilon = ck["synced"], ck["updates"], ck["epsilon"]
    agent.total_stacks, agent.stacks_since_sync = ck["total_stacks"], ck["stacks_since_sync"]
    torch.set_rng_state(ck["torch_rng"])
    with open(path / "rest.pkl", "rb") as f:
        rest = pickle.load(f)
    rp.tree, rp.action, rp.reward, rp.done = rest["tree"], rest["action"], rest["reward"], rest["done"]
    rp.terminal_next = rest["terminal_next"]
    agent.rng.setstate(rest["rng"])
    agent.np_rng.bit_generator.state = rest["np_rng"]
    random.setstate(rest["py_rng"])
    return ck["episode"], ck["cum_ticks"], ck["wall"]


def evaluate(env: GvgaiEnv, agent: Agent2018, games: int) -> dict:
    net = agent.q
    if agent.synced_state is not None:
        net = type(agent.q)(*agent.replay.state.shape[2:], agent.n_actions).to(agent.device)
        net.load_state_dict(agent.synced_state)
    rows = [play_episode(env, agent, train=False, greedy=True, net=net) for _ in range(games)]
    return {
        "games": games,
        "mean_score": float(np.mean([r["score"] for r in rows])),
        "win_rate": float(np.mean([r["win"] for r in rows])),
        "scores": [r["score"] for r in rows],
        "wins": [r["win"] for r in rows],
        "levels": [r["level"] for r in rows],
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="aliens")
    p.add_argument("--episodes", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--per-sample", action="store_true", help="2018's one-at-a-time forward passes")
    p.add_argument("--eval-games", type=int, default=50)
    p.add_argument("--checkpoint-minutes", type=float, default=15)
    p.add_argument("--warmup-stacks", type=int, default=Config.warmup_stacks, help="smaller only for tests")
    p.add_argument("--target-update-stacks", type=int, default=Config.target_update_stacks)
    p.add_argument("--loss-penalty", type=float, default=Config.loss_penalty, help="0 = 2018's old reward")
    p.add_argument("--fix-bugs", action="store_true", help="fix 2018's reward timing and PER weights")
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    run = HERE / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    ckpt = run / "checkpoint"
    random.seed(a.seed)
    env = GvgaiEnv(a.game, include_nil=True, engine_seed=a.seed + 1)
    obs, _ = env.reset()
    from agents.ddqn2018 import preprocess

    h, w = preprocess(obs).shape
    cfg = Config(
        seed=a.seed,
        per_sample=a.per_sample,
        warmup_stacks=a.warmup_stacks,
        target_update_stacks=a.target_update_stacks,
        loss_penalty=a.loss_penalty,
        fix_bugs=a.fix_bugs,
    )
    agent = Agent2018(env.n_actions - 1, (h, w), cfg, a.device)

    episode, cum_ticks, wall0 = 0, 0, 0.0
    if ckpt.exists():
        episode, cum_ticks, wall0 = load_checkpoint(ckpt, agent)
        print(f"resumed at episode {episode}")
    csv_path = run / "episodes.csv"
    if episode == 0:
        csv_path.write_text(",".join(COLUMNS) + "\n")
    else:  # drop rows written after the checkpoint
        lines = csv_path.read_text().splitlines()[: episode + 1]
        csv_path.write_text("\n".join(lines) + "\n")

    t_start = time.perf_counter() - wall0
    last_ckpt = time.perf_counter()
    with open(csv_path, "a", newline="") as f:
        out = csv.writer(f)
        while episode < a.episodes:
            row = play_episode(env, agent)
            episode += 1
            cum_ticks += row["ticks"]
            wall = time.perf_counter() - t_start
            out.writerow([episode, row["score"], row["total_reward"], row["steps"], row["win"],
                          f"{row['epsilon']:.4f}", row["ticks"], row["trained"], row["warmup"], row["level"],
                          cum_ticks, f"{wall:.1f}"])  # fmt: skip
            f.flush()
            if episode % 50 == 0:
                print(
                    f"ep {episode} eps {agent.epsilon:.3f} ticks {cum_ticks} wall {wall / 60:.1f} min",
                    flush=True,
                )
            if time.perf_counter() - last_ckpt > a.checkpoint_minutes * 60:
                save_checkpoint(ckpt, agent, episode, cum_ticks, wall)
                last_ckpt = time.perf_counter()

    train_wall = time.perf_counter() - t_start
    t_eval = time.perf_counter()
    ev = evaluate(env, agent, a.eval_games)
    env.close()
    result = {
        "name": a.name, "agent": "ddqn2018-fixed" if a.fix_bugs else "ddqn2018", "game": a.game, "seed": a.seed,
        "episodes": a.episodes,
        "train_ticks": cum_ticks, "train_wall_s": train_wall, "eval_wall_s": time.perf_counter() - t_eval,
        "updates": agent.updates, "device": a.device, "per_sample": a.per_sample, "config": asdict(cfg),
        "eval": ev, "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                             text=True).stdout.strip(),
        "machine": platform.processor(), "torch": torch.__version__,
    }  # fmt: skip
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{a.name}.json").write_text(json.dumps(result, indent=1))
    ev_s = f"{ev['mean_score']:.1f} ({ev['win_rate']:.0%})"
    print(f"done: {cum_ticks} ticks, {train_wall / 60:.1f} min; eval {ev_s}")


if __name__ == "__main__":
    main()
    os._exit(0)  # see train_modern.py: teardown can segfault on Windows after everything is written
