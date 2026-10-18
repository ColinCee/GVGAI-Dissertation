"""Play the engine's sampleMCTS on each game with 2018's limits (40 ms per move, 2000 ticks).

    uv run python run_mcts.py --games aliens chase --per-level 17

One JVM per game, in parallel. Writes runs/mcts/<game>.csv and results/mcts-<game>.json.
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

GAMES = ["aliens", "chase", "deceptizelda", "waferthinmints2018"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", default=GAMES)
    p.add_argument("--per-level", type=int, default=17)
    p.add_argument("--ms", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    classes = build.ensure_built()
    out = HERE / "runs" / "mcts"
    out.mkdir(parents=True, exist_ok=True)
    procs = {}
    for g in a.games:
        d = HERE / "engine" / "games" / f"{g}_v0"
        cmd = [build.java(), "-XX:+UseSerialGC", "-Xmx512m", f"-Dgvgai.actionTime={a.ms}", "-Dgvgai.maxTicks=2000",
               "-cp", str(classes), "tracks.RunMCTS", str(d / f"{g}.txt"), str(d / f"{g}_lvl"), "0,1,2",
               str(a.per_level), str(a.seed)]  # fmt: skip
        procs[g] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    for g, pr in procs.items():
        text, _ = pr.communicate()
        lines = [ln for ln in text.splitlines() if ln[:1].isdigit() or ln.startswith("level,")]
        (out / f"{g}.csv").write_text("\n".join(lines) + "\n")
        rows = list(csv.DictReader(io.StringIO("\n".join(lines))))[:50]
        res = {
            "name": f"mcts-{g}", "agent": f"MCTS {a.ms} ms", "game": g, "seed": a.seed,
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
