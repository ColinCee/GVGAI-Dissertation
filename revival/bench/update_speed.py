"""Milliseconds per learner update on a filled replay (random data), for a preset.

uv run python bench/update_speed.py dqn rainbow
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.modern import ModernAgent  # noqa: E402
from train_modern import PRESETS  # noqa: E402

if __name__ == "__main__":
    for preset in sys.argv[1:] or ["dqn"]:
        n, h, w = 8, 55, 150
        agent = ModernAgent(n, h, w, 3, PRESETS[preset](), "cuda", 0)
        rng = np.random.default_rng(0)
        for _ in range(5000):
            obs = rng.integers(0, 255, (n, h, w), dtype=np.uint8)
            term = rng.random(n) < 0.002
            agent.replay.add(obs, rng.integers(0, 3, n), rng.integers(-1, 2, n).astype(np.float32), term,
                             np.zeros(n, bool), obs)  # fmt: skip
        for _ in range(50):
            agent.update(0.5)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(500):
            agent.update(0.5)
        torch.cuda.synchronize()
        print(f"{preset}: {(time.perf_counter() - t0) / 500 * 1000:.2f} ms per update", flush=True)
