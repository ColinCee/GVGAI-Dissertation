"""Pick each game's best 5M-tick setup from the finished runs and queue it at 10M ticks, 3 seeds, 100 eval games.

    uv run python bestof.py            # writes jobs/queue/20-bestof.txt once every queued run has a result
    uv run python bestof.py --force    # pick from whatever has finished

Candidates: run groups trained to 5M ticks ("-long"), with at least 2 seeds, whose seed-0 command is in a job file.
Ranked by mean win rate on the training levels, then mean score (Chase: nobody wins, so score decides).
"""

import argparse
import re
from collections import defaultdict

from report import GAMES, HERE, load

JOBS = HERE / "jobs"
OUT = JOBS / "queue" / "20-bestof.txt"


def job_lines() -> dict[str, str]:
    lines = {}
    for f in [*sorted(JOBS.glob("*.txt")), *sorted((JOBS / "queue").glob("*.txt"))]:
        if f == OUT:
            continue
        for ln in f.read_text().splitlines():
            if ln.strip() and not ln.startswith("#") and "--name" in ln:
                lines[ln.split("--name")[1].split()[0]] = ln.strip()
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry", action="store_true", help="print the picks without queueing them")
    a = p.parse_args()
    lines = job_lines()
    missing = [n for n in lines if not (HERE / "results" / f"{n}.json").exists()]
    if missing and not a.force:
        print(f"{len(missing)} queued runs without results, e.g. {missing[0]}")
        return
    groups = load()
    picks = defaultdict(list)
    for key, runs in groups.items():
        g = runs[0].get("game")
        if g in GAMES and key.endswith("-long") and len(runs) >= 2 and f"{key}-s0" in lines:
            win = sum(r["eval"]["win_rate"] for r in runs) / len(runs)
            score = sum(r["eval"]["mean_score"] for r in runs) / len(runs)
            picks[g].append((win, score, key))
    out = ["# Best 5M setup per game, at 10M ticks, 3 seeds, 100 eval games (written by bestof.py)"]
    for g in GAMES:
        if not picks[g]:
            continue
        win, score, key = max(picks[g])
        base = lines[f"{key}-s0"]
        out.append(f"# {g}: {key} ({score:.1f}, {win:.0%} at 5M)")
        for seed in range(3):
            ln = re.sub(r"--ticks \d+", "--ticks 10000000", base)
            ln = re.sub(r"--seed \d+", f"--seed {seed}", ln)
            ln = re.sub(r"--name \S+", f"--name {g}-best-s{seed}", ln)
            out.append(ln + " --eval-games 100")
    if not a.dry:
        OUT.write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
