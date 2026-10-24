"""Replay a trained run's evaluation and report how each lost game ended.

    uv run python -m tools.losses aliens-dqn-outcome-score01-ar1-e100-s0 [--heldout] [--device cpu]

Rebuilds the agent from results/<name>.json, loads runs/<name>/q.pt and plays the same games as the
run's evaluation (same seed, levels and number of games). For each game: level, score, ticks, and
how it ended (win, death, or the 2000-tick timeout). The last frames of each lost game are saved to
runs/<name>/lost-<i>.png (newest on the right) to look at.
"""

import argparse
import json
from dataclasses import fields

import numpy as np
import torch
from PIL import Image

from agents.evaluate import Stacks
from agents.modern import ModernAgent, ModernConfig
from gvgai.games import HELD_OUT, ROOT, game_levels
from gvgai.vec import vector

KEEP = 12  # last frames kept per game


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("name")
    p.add_argument("--heldout", action="store_true")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    res = json.loads((ROOT / "results" / f"{a.name}.json").read_text())
    names = {f.name for f in fields(ModernConfig)}
    cfg = ModernConfig(**{k: v for k, v in res["config"].items() if k in names})
    cfg.compile = False
    game, seed, envs = res["game"], res["seed"], res["envs"]
    ev = res["eval_heldout" if a.heldout else "eval"]
    levels = HELD_OUT if a.heldout else game_levels(game)
    env = vector(game, envs, seed + (8 if a.heldout else 7), levels, cfg.action_repeat)
    obs, _ = env.reset(seed=seed + (8 if a.heldout else 7))
    h, w = obs.shape[1:]
    agent = ModernAgent(envs, h, w, int(env.single_action_space.n), cfg, a.device, seed)
    agent.q.load_state_dict(torch.load(ROOT / "runs" / a.name / "q.pt", map_location=a.device))
    agent.q.eval()
    stacks = Stacks(obs, agent.device)
    games, per_env = ev["games"], -(-ev["games"] // envs)
    done = np.zeros(envs, np.int64)
    ticks = np.zeros(envs, np.int64)
    deaths = np.zeros(envs, np.int64)
    recent: list[list[np.ndarray]] = [[o] for o in obs]
    rows = []
    with torch.no_grad():
        while (done < per_env).any():
            act = agent.act(stacks.array(), 0.0)
            obs, rew, term, trunc, info = env.step(act)
            ticks += info["ticks"]
            deaths += rew < 0  # Aliens: -1 when the ship is hit
            ended = term | trunc
            for e in range(envs):
                if not ended[e]:
                    recent[e] = (recent[e] + [obs[e]])[-KEEP:]
            for e in np.flatnonzero(ended):
                fi = info["final_info"]
                if done[e] < per_env:
                    win = bool(fi["win"][e])
                    how = "win" if win else ("timeout" if trunc[e] else "death")
                    rows.append({"level": int(fi["level"][e]), "score": float(fi["score"][e]), "ticks": int(ticks[e]),
                                 "how": how, "frames": recent[e] + [info["final_obs"][e]]})  # fmt: skip
                done[e] += 1
                ticks[e] = deaths[e] = 0
                recent[e] = [obs[e]]
            stacks.push(obs, ended)
    env.close()
    rows = rows[:games]
    same = [int(r["how"] == "win") for r in rows] == ev["wins"]
    print(f"{a.name} {'held-out' if a.heldout else 'training'} levels: {sum(r['how'] == 'win' for r in rows)}/{len(rows)} won"
          f" (matches the run's own evaluation: {same})")  # fmt: skip
    wins = [r for r in rows if r["how"] == "win"]
    print(f"wins: mean {np.mean([r['ticks'] for r in wins]):.0f} ticks, longest {max(r['ticks'] for r in wins)}")
    for i, r in enumerate(rows):
        if r["how"] != "win":
            strip = np.concatenate(r["frames"], axis=1)
            Image.fromarray(strip).save(ROOT / "runs" / a.name / f"lost-{i}.png")
            print(f"lost game {i}: level {r['level']}, {r['how']} at tick {r['ticks']}, score {r['score']:g}")


if __name__ == "__main__":
    main()
