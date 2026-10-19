"""Ticks per second of the 2018 agent training on Aliens: 400 games (past the 20k-stack warm-up).

uv run python bench/train2018_speed.py [cpu|cuda]
"""

import csv
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    name = f"bench-2018-{device}"
    subprocess.run([sys.executable, "-u", "-m", "train.ddqn2018", "--game", "aliens", "--episodes", "400",
                    "--seed", "0", "--eval-games", "2", "--device", device, "--threads", "1", "--name", name],
                   cwd=HERE, capture_output=True, check=False)  # fmt: skip
    rows = list(csv.DictReader((HERE / "runs" / name / "episodes.csv").read_text().splitlines()))
    trained = [r for r in rows if r["warmup"] == "0"]
    first, last = trained[0], trained[-1]
    rate = (int(last["cum_ticks"]) - int(first["cum_ticks"])) / (float(last["wall_s"]) - float(first["wall_s"]))
    warm = int(rows[-1]["cum_ticks"]) and int(first["cum_ticks"]) / float(first["wall_s"])
    (HERE / "results" / f"{name}.json").unlink(missing_ok=True)
    for f in (HERE / "runs" / name).glob("**/*"):
        if f.is_file():
            f.unlink()
    print(f"2018 agent on {device}: warm-up {warm:,.0f} ticks/s, training {rate:,.0f} ticks/s", flush=True)
