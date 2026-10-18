"""Ticks per second for one game: random play, 4-tick action repeat, fixed seeds, after a warm-up.

    uv run python bench/game_speed.py [--full-obs] [--png]
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgai.env import GvgaiEnv  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--game", default="aliens")
p.add_argument("--seconds", type=float, default=30)
p.add_argument("--png", action="store_true", help="old PNG frame path")
p.add_argument("--full-obs", action="store_true", help="per-tick sprite copies (light observation off)")
a = p.parse_args()

env = GvgaiEnv(a.game, engine_seed=1, raw_frames=not a.png, light_obs=not a.full_obs)
rng = random.Random(1)


def run(seconds: float) -> float:
    ticks, t0 = 0, time.perf_counter()
    env.reset()
    t = 0
    while time.perf_counter() - t0 < seconds:
        if t % 4 == 0:
            act = rng.randrange(env.n_actions)
        t += 1
        *_, term, trunc, _ = env.step(act)
        ticks += 1
        if term or trunc:
            env.reset()
            t = 0
    return ticks / (time.perf_counter() - t0)


run(5)  # JIT warm-up
print(f"{a.game}: {run(a.seconds):.0f} ticks/s")
env.close()
