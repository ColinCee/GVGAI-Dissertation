"""Ticks per second of a real training run: train/modern.py on Aliens, 300k ticks, 8 games.

uv run python bench/train_speed.py dqn rainbow
"""

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    for preset in sys.argv[1:] or ["dqn"]:
        name = f"bench-train-{preset}"
        out = subprocess.run([sys.executable, "-u", "-m", "train.modern", "--game", "aliens", "--preset", preset,
                              "--ticks", "300000", "--envs", "8", "--eval-games", "4", "--seed", "0", "--name", name],
                             cwd=HERE, capture_output=True, text=True).stdout  # fmt: skip
        rate = re.findall(r"^ticks \d+ \((\d+)/s\).*", out, re.M)
        (HERE / "results" / f"{name}.json").unlink(missing_ok=True)  # a benchmark, not a result
        print(f"{preset}: {rate[-1] if rate else 'FAILED'} ticks/s while training", flush=True)
