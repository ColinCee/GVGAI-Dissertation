"""Play the engine's sampleMCTS on each game with 2018's limits (40 ms per move, 2000 ticks).

    uv run python run_mcts.py --games aliens chase

One JVM per game and level, all at once (12 on 16 threads): Chase alone took ~1 h in one process.
Same levels as the agents (Deceptizelda 0 and 2), 50 games per game split over its levels.
Writes runs/mcts/<game>.csv and results/mcts-<game>.json.
"""

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "engine"))
import build  # noqa: E402
from train2018 import DEFAULT_LEVELS  # noqa: E402

GAMES = ["aliens", "chase", "deceptizelda", "waferthinmints2018"]
HEADER = "level,score,win,ticks,iters_per_move"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", default=GAMES)
    p.add_argument("--games-per-game", type=int, default=50)
    p.add_argument("--ms", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    classes = build.ensure_built()
    out = HERE / "runs" / "mcts"
    out.mkdir(parents=True, exist_ok=True)
    procs: dict[str, list[subprocess.Popen]] = {}
    for g in a.games:
        d = HERE / "engine" / "games" / f"{g}_v0"
        levels = DEFAULT_LEVELS.get(g, (0, 1, 2))
        per_level = -(-a.games_per_game // len(levels))
        procs[g] = []
        for lvl in levels:
            cmd = [build.java(), "-XX:+UseSerialGC", "-Xmx512m", f"-Dgvgai.actionTime={a.ms}", "-Dgvgai.maxTicks=2000",
                   "-cp", str(classes), "tracks.RunMCTS", str(d / f"{g}.txt"), str(d / f"{g}_lvl"), str(lvl),
                   str(per_level), str(a.seed + lvl)]  # fmt: skip
            procs[g].append(subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True))
    for g, prs in procs.items():
        lines = [HEADER]
        for pr in prs:
            text, _ = pr.communicate()
            lines += [ln for ln in text.splitlines() if ln[:1].isdigit()]
        (out / f"{g}.csv").write_text("\n".join(lines) + "\n")
        rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
        if not rows:
            print(f"{g}: no games finished")
            continue
        res = {
            "name": f"mcts-{g}", "agent": f"MCTS {a.ms} ms", "game": g, "seed": a.seed,
            "levels": list(DEFAULT_LEVELS.get(g, (0, 1, 2))),
            "eval": {
                "games": len(rows),
                "mean_score": sum(float(r["score"]) for r in rows) / len(rows),
                "win_rate": sum(int(r["win"]) for r in rows) / len(rows),
                "iters_per_move": sum(float(r["iters_per_move"]) for r in rows) / len(rows),
            },
        }  # fmt: skip
        (HERE / "results" / f"mcts-{g}.json").write_text(json.dumps(res, indent=1))
        e = res["eval"]
        print(f"{g}: {e['mean_score']:.1f} ({e['win_rate']:.0%}), {e['iters_per_move']:.0f} iterations per move")


if __name__ == "__main__":
    main()
