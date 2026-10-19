"""Build the comparison tables from results/*.json and the 2018 data.

    uv run python report.py            # writes results/summary.md

Runs are grouped by name without the seed suffix; seeds are averaged, win-% range shown.
Superseded runs (results/superseded/) are left out.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SHEET = Path(r"D:\Claude Projects\Dissertation-Report\spreadsheet")
GAMES = ["aliens", "chase", "deceptizelda", "waferthinmints2018"]
NAMES = {"aliens": "Aliens", "chase": "Chase", "deceptizelda": "Deceptizelda",
         "waferthinmints2018": "Wafer Thin Mints"}  # fmt: skip

# The 2018 report's final table: (score, win %); 50 games each, human 10.
TABLE_2018 = {
    "Human (2018)": {"aliens": (42.6, 100), "chase": (1.2, 0), "deceptizelda": (10.6, 80),
                     "waferthinmints2018": (11, 0)},
    "MCTS, 40 ms (2018)": {"aliens": (79.2, 100), "chase": (3.5, 3.7), "deceptizelda": (1.91, 84.4),
                           "waferthinmints2018": (9.9, 75.5)},
    "2018 final agent (2018 report)": {"aliens": (56.3, 82.6), "chase": (0.7, 0), "deceptizelda": (1.99, 99),
                                       "waferthinmints2018": (10.2, 33)},
}  # fmt: skip

# Rows of the main table: label -> run-name pattern ({g} = game). "b18" = 2018's ticks for that game.
ROWS_2026 = [
    ("MCTS, 40 ms (2026 PC)", "mcts-{g}"),
    ("2018 final agent, replayed", "{g}-ddqn2018"),
    ("2018 final agent, bugs fixed", "{g}-ddqn2018fixed"),
    ("Modern DQN, 2018's ticks", "{g}-dqn-b18"),
    ("Modern DQN + 2018's reward, 2018's ticks", "{g}-dqn-reward2018-b18"),
    ("Rainbow, 2018's ticks", "{g}-rainbow-b18"),
    ("Rainbow, IMPALA network, 2018's ticks", "{g}-rainbow-impala-b18"),
    ("PPO, 2018's ticks", "{g}-ppo-b18"),
    ("Modern DQN, up to 5M ticks", "{g}-dqn-long"),
    ("Rainbow, up to 5M ticks", "{g}-rainbow-long"),
    ("Best 5M setup, up to 10M ticks, 100 eval games", "{g}-best"),
]

# 2018 report: Aliens win % per 500 training games for each variant.
CURVES_2018 = {
    "DQN": [6, 12, 29, 35],
    "DDQN": [3.0, 17.4, 47.4, 56.6],
    "DDQN + PER": [3.4, 12.6, 31.8, 54.2],
    "DDQN + PER + new reward (final)": [4.0, 22.2, 58.0, 68.2],
}
ABLATION_2026 = {
    "DQN": "aliens-ddqn2018-dqn",
    "DDQN": "aliens-ddqn2018-ddqn",
    "Dueling": "aliens-ddqn2018-dueling",
    "DDQN + PER + new reward (final)": "aliens-ddqn2018",
    "Final, bugs fixed": "aliens-ddqn2018fixed",
}
LADDER = [
    ("2018 final agent, replayed", "aliens-ddqn2018"),
    ("Modern DQN (Double, Huber, Adam, /255 inputs)", "aliens-dqn-b18"),
    ("+ prioritised replay", "aliens-dqn-per"),
    ("+ dueling", "aliens-dqn-per-dueling"),
    ("+ 3-step returns", "aliens-dqn-per-dueling-nstep"),
    ("+ noisy nets + distributional = Rainbow", "aliens-rainbow-b18"),
]


def read_csv(path: Path, header: bool = True) -> list:
    lines = path.read_text().splitlines()
    return list(csv.DictReader(lines)) if header else [r for r in csv.reader(lines) if r]


def load() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for p in sorted(RESULTS.glob("*.json")):
        d = json.loads(p.read_text())
        if "eval" not in d or p.stem.startswith(("bench-", "ab-", "smoke", "prof", "dbg", "speed", "con-")):
            continue
        key = p.stem.rsplit("-s", 1)[0] if p.stem.rsplit("-s", 1)[-1].isdigit() else p.stem
        groups[key].append(d)
    return groups


def cell(runs: list[dict], key: str = "eval") -> str:
    if not runs:
        return "–"
    s = np.array([r[key]["mean_score"] for r in runs])
    w = 100 * np.array([r[key]["win_rate"] for r in runs])
    if len(runs) == 1:
        return f"{s[0]:.1f} ({w[0]:.0f}%)"
    return f"{s.mean():.1f} ({w.mean():.0f}%, {w.min():.0f}–{w.max():.0f})"


def win_blocks(wins: list[int], size: int = 500, upto: int = 2000) -> list[float]:
    return [100 * float(np.mean(wins[a : a + size])) for a in range(0, upto, size) if wins[a : a + size]]


def main() -> None:
    g = load()
    out = ["# Results", "",
           "Average score (win %) over 50 evaluation games, greedy, levels 0–2 (Deceptizelda 0 and 2). For 2026 runs: "
           "mean over 3 seeds, then the range of win % across seeds; 2018 rows are single runs.", ""]  # fmt: skip
    out += ["| Agent | " + " | ".join(NAMES[x] for x in GAMES) + " |", "|---" * (len(GAMES) + 1) + "|"]
    for label, row in TABLE_2018.items():
        out.append(f"| {label} | " + " | ".join(f"{row[x][0]} ({row[x][1]}%)" for x in GAMES) + " |")
    for label, pat in ROWS_2026:
        out.append(f"| {label} | " + " | ".join(cell(g.get(pat.format(g=x), [])) for x in GAMES) + " |")

    out += ["", "## Held-out levels 3–4 (never trained on)", "",
            "| Agent | " + " | ".join(NAMES[x] for x in GAMES) + " |", "|---" * (len(GAMES) + 1) + "|"]  # fmt: skip
    for label, pat in ROWS_2026[3:]:
        runs = {x: [r for r in g.get(pat.format(g=x), []) if "eval_heldout" in r] for x in GAMES}
        out.append(f"| {label} | " + " | ".join(cell(runs[x], "eval_heldout") for x in GAMES) + " |")

    out += ["", "## 2018's ablation on Aliens: win % per 500 training games", "",
            "| Variant | 2018 report | 2026 replay (mean of seeds) | 2026 eval score (wins) |",
            "|---|---|---|---|"]  # fmt: skip
    for name, run in ABLATION_2026.items():
        rep = CURVES_2018.get(name)
        rep_s = " / ".join(f"{b:.0f}" for b in rep) if rep else "– (figure only)"
        curves = [win_blocks([int(x["win"]) for x in read_csv(HERE / "runs" / r["name"] / "episodes.csv")])
                  for r in g.get(run, []) if (HERE / "runs" / r["name"] / "episodes.csv").exists()]  # fmt: skip
        cur_s = " / ".join(f"{b:.0f}" for b in np.mean(curves, axis=0)) if curves else "–"
        out.append(f"| {name} | {rep_s} | {cur_s} | {cell(g.get(run, []))} |")

    out += ["", "## Improvement ladder on Aliens, 2018's ticks (1.01M), seed 0 unless noted", "",
            "| Step | Eval score (wins) | Train time |", "|---|---|---|"]  # fmt: skip
    for label, run in LADDER:
        runs = g.get(run, [])
        t = f"{np.mean([r['train_wall_s'] for r in runs]) / 60:.1f} min" if runs else "–"
        out.append(f"| {label} | {cell(runs)} | {t} |")

    out += ["", "## Training time for one run", "", "| Run | Game ticks | Wall clock | Ticks/s |", "|---|---|---|---|",
            "| 2018 plain DQN on Aliens, GTX 980 (report) | ~1.0M | ~12 h | ~23 |"]  # fmt: skip
    for label, run in [("2018 final agent replayed, Aliens", "aliens-ddqn2018"), ("Modern DQN, Aliens, 2018's ticks",
                       "aliens-dqn-b18"), ("Rainbow, Aliens, 2018's ticks", "aliens-rainbow-b18"),
                       ("Modern DQN, Aliens, 5M ticks", "aliens-dqn-long"), ("Rainbow, Aliens, 5M ticks",
                       "aliens-rainbow-long")]:  # fmt: skip
        runs = g.get(run, [])
        if runs:
            t = np.mean([r["train_ticks"] for r in runs])
            s = np.mean([r["train_wall_s"] for r in runs])
            out.append(f"| {label} (mean of {len(runs)}) | {t / 1e6:.2f}M | {s / 60:.0f} min | {t / s:,.0f} |")

    out += [
        "",
        "## Every run group",
        "",
        "| Runs | Seeds | Eval score (wins) | Held-out levels | Ticks | Wall clock |",
        "|---|---|---|---|---|---|",
    ]
    for k in sorted(g):
        runs = [r for r in g[k] if "train_ticks" in r]
        if runs:
            held = cell([r for r in runs if "eval_heldout" in r], "eval_heldout")
            t, s = np.mean([r["train_ticks"] for r in runs]), np.mean([r["train_wall_s"] for r in runs])
            out.append(f"| {k} | {len(runs)} | {cell(runs)} | {held} | {t / 1e6:.2f}M | {s / 60:.0f} min |")

    stopped = [(k, r) for k, runs in g.items() for r in runs if r.get("stopped_early_at")]
    if stopped:
        out += ["", "Long runs stopped early (no longer improving after 3M ticks): "
                + ", ".join(f"{r['name']} at {r['stopped_early_at'] / 1e6:.1f}M" for _, r in stopped)]  # fmt: skip
    (RESULTS / "summary.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
