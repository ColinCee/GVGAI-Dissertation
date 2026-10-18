"""Play the engine's sampleMCTS on each game with 2018's limits (40 ms per move, 2000 ticks).

    uv run python run_mcts.py --games aliens chase

One JVM per game (4 at once, one per core, so the 40 ms budget isn't shared); --split-levels runs one
per level instead.
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
    p.add_argument("--iters", type=int, default=0, help="fixed search budget per move instead of --ms")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--split-levels", action="store_true", help="one process per level (faster, but more processes than cores)"
    )
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
        for lvl in levels if a.split_levels else [",".join(map(str, levels))]:
            budget = f"-Dgvgai.actionTime={a.ms if not a.iters else 10**8}"
            cmd = [build.java(), "-XX:+UseSerialGC", "-Xmx512m", budget, f"-Dgvgai.mctsIters={a.iters}",
                   "-Dgvgai.maxTicks=2000", "-cp", str(classes), "tracks.RunMCTS", str(d / f"{g}.txt"),
                   str(d / f"{g}_lvl"), str(lvl), str(per_level), str(a.seed)]  # fmt: skip
            # cwd: the engine loads sprites by relative path
            procs[g].append(
                subprocess.Popen(cmd, cwd=HERE / "engine", stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            )
    for g, prs in procs.items():
        lines = [HEADER]
        for pr in prs:
            text, _ = pr.communicate()
            lines += [ln for ln in text.splitlines() if ln[:1].isdigit()]
        (out / f"{g}{'-i' + str(a.iters) if a.iters else ''}.csv").write_text("\n".join(lines) + "\n")
        rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
        if not rows:
            print(f"{g}: no games finished")
            continue
        res = {
            "name": f"mcts-{g}" + (f"-i{a.iters}" if a.iters else ""),
            "agent": f"MCTS {a.iters} iterations" if a.iters else f"MCTS {a.ms} ms", "game": g, "seed": a.seed,
            "levels": list(DEFAULT_LEVELS.get(g, (0, 1, 2))),
            "eval": {
                "games": len(rows),
                "mean_score": sum(float(r["score"]) for r in rows) / len(rows),
                "win_rate": sum(int(r["win"]) for r in rows) / len(rows),
                "iters_per_move": sum(float(r["iters_per_move"]) for r in rows) / len(rows),
            },
        }  # fmt: skip
        (HERE / "results" / f"{res['name']}.json").write_text(json.dumps(res, indent=1))
        e = res["eval"]
        print(f"{g}: {e['mean_score']:.1f} ({e['win_rate']:.0%}), {e['iters_per_move']:.0f} iterations per move")


if __name__ == "__main__":
    main()
