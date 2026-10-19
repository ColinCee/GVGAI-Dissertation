"""Quick screen: a few setups, 2 seeds each, a short budget, all at once; then one table to pick winners from.

    uv run python screen.py dqn dqn-outcome-only ppo ppo:--outcome-only --ticks 1000000
    uv run python screen.py dqn-outcome-only dqn-outcome-score01 --game aliens --ticks 5000000 --seeds 3

A setup is a train_modern.py preset, or "ppo" / "ppo:<train_ppo.py flags>". Runs are named
<game>-<setup>-scr<ticks>-s<seed> and skipped when already done, so a screen can be extended later.
Ranked on the training curve (wins and score over the last quarter of training games, averaged over seeds),
which uses far more games than the 50-game evaluation; the evaluation is shown too.
"""

import argparse
import csv
import json
import re
import subprocess
import sys

import numpy as np

from report import HERE


def job(game: str, setup: str, ticks: int, seed: int) -> tuple[str, str]:
    tag = re.sub(r"[^a-z0-9.]+", "-", setup.replace("ppo:--", "ppo-").lower()).strip("-")
    name = f"{game}-{tag}-scr{ticks / 1e6:g}m-s{seed}"
    if setup == "ppo" or setup.startswith("ppo:"):
        flags = setup.split(":", 1)[1] if ":" in setup else ""
        return name, f"train_ppo.py --game {game} --ticks {ticks} --seed {seed} {flags} --name {name}"
    return name, f"train_modern.py --game {game} --preset {setup} --ticks {ticks} --seed {seed} --name {name}"


def curve(name: str) -> tuple[float, float]:
    rows = list(csv.DictReader((HERE / "runs" / name / "episodes.csv").read_text().splitlines()))
    tail = rows[-max(1, len(rows) // 4) :]
    return float(np.mean([float(r["score"]) for r in tail])), 100 * float(np.mean([int(r["win"]) for r in tail]))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("setups", nargs="+")
    p.add_argument("--game", default="aliens")
    p.add_argument("--ticks", type=int, default=1_000_000)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--parallel", type=int, default=4)
    a = p.parse_args()
    jobs = {s: [job(a.game, s, a.ticks, k) for k in range(a.seeds)] for s in a.setups}
    path = HERE / "runs" / "screens" / f"{a.game}-{a.ticks}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(cmd for js in jobs.values() for _, cmd in js) + "\n")
    # no check: failed runs show as "failed" in the table
    subprocess.run([sys.executable, "-u", "batch.py", str(path), "--parallel", str(a.parallel)], cwd=HERE)
    print(f"\n{a.game}, {a.ticks / 1e6:g}M ticks, {a.seeds} seeds. Last quarter of training | evaluation (50 games)")
    print("| Setup | Training score | Training wins | Eval score | Eval wins | Minutes per run |")
    print("|---|---|---|---|---|---|")
    for s, js in jobs.items():
        done = [n for n, _ in js if (HERE / "results" / f"{n}.json").exists()]
        if not done:
            print(f"| {s} | failed | | | | |")
            continue
        c = np.array([curve(n) for n in done])
        r = [json.loads((HERE / "results" / f"{n}.json").read_text()) for n in done]
        ev_s = np.mean([x["eval"]["mean_score"] for x in r])
        ev_w = 100 * np.mean([x["eval"]["win_rate"] for x in r])
        mins = np.mean([x["train_wall_s"] for x in r]) / 60
        print(f"| {s} | {c[:, 0].mean():.1f} | {c[:, 1].mean():.0f}% | {ev_s:.1f} | {ev_w:.0f}% | {mins:.1f} |")


if __name__ == "__main__":
    main()
