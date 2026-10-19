"""Ticks per second through the modern agents' env layer (gvgai/vec.py), random moves.

uv run python bench/env_layer_speed.py 8 16
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgai.vec import vector  # noqa: E402

if __name__ == "__main__":
    for n in [int(x) for x in sys.argv[1:]] or [8]:
        env = vector("aliens", n, seed=1)
        env.reset(seed=1)
        rng = np.random.default_rng(0)
        for _ in range(50):
            env.step(rng.integers(0, 3, n))
        t0, ticks = time.perf_counter(), 0
        while time.perf_counter() - t0 < 20:
            *_, info = env.step(rng.integers(0, 3, n))
            ticks += int(info["ticks"].sum())
        print(f"{n} games: {ticks / (time.perf_counter() - t0):,.0f} ticks/s", flush=True)
        env.close()
