"""Build the comparison tables from results/*.json and the 2018 data.

    uv run python report.py            # writes results/summary.md

Rows are grouped by agent and game; seeds are averaged and the spread shown as min-max.
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
GAME_NAMES = {"aliens": "Aliens", "chase": "Chase", "deceptizelda": "Deceptizelda",
              "waferthinmints2018": "Wafer Thin Mints"}  # fmt: skip

# From the 2018 report's final table (score, win %); 50 games each, human 10.
TABLE_2018 = {
    "Human (2018)": {"aliens": (42.6, 100), "chase": (1.2, 0), "deceptizelda": (10.6, 80),
                     "waferthinmints2018": (11, 0)},
    "MCTS (2018)": {"aliens": (79.2, 100), "chase": (3.5, 3.7), "deceptizelda": (1.91, 84.4),
                    "waferthinmints2018": (9.9, 75.5)},
    "DDQN+PER (2018)": {"aliens": (56.3, 82.6), "chase": (0.7, 0), "deceptizelda": (1.99, 99),
                        "waferthinmints2018": (10.2, 33)},
}  # fmt: skip


def read_csv(path: Path, header: bool = True) -> list:
    lines = path.read_text().splitlines()
    return list(csv.DictReader(lines)) if header else [r for r in csv.reader(lines) if r]


def win_blocks(wins: list[int], size: int = 500, upto: int = 2000) -> list[float]:
    return [100 * float(np.mean(wins[a : a + size])) for a in range(0, upto, size) if wins[a : a + size]]


def curves_2018() -> dict[str, list[float]]:
    out = {}
    rows = read_csv(RESULTS / "2018" / "aliens-ddqn-per-reward.csv")
    out["2018 run A (report)"] = win_blocks([int(r["win"]) for r in rows])
    if (SHEET / "DDQN + PER Reward.csv").exists():
        rows2 = read_csv(SHEET / "DDQN + PER Reward.csv", header=False)
        out["2018 run B (CSV)"] = win_blocks([int(float(r[2])) for r in rows2])
    return out


def load() -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]


def cell(vals: list[tuple[float, float]]) -> str:
    s, w = np.array([v[0] for v in vals]), np.array([v[1] for v in vals])
    if len(vals) == 1:
        return f"{s[0]:.1f} ({w[0]:.0f}%)"
    return f"{s.mean():.1f} ({w.mean():.0f}%, {w.min():.0f}–{w.max():.0f}, n={len(vals)})"


def main() -> None:
    res = load()
    lines = ["# Results", "", "Average score (win %) over 50 evaluation games, levels 0–2. "
             "For our runs: mean over seeds, win-% range, seed count.", ""]  # fmt: skip
    lines += ["| Agent | " + " | ".join(GAME_NAMES[g] for g in GAMES) + " |", "|---" * (len(GAMES) + 1) + "|"]
    for name, row in TABLE_2018.items():
        lines.append(f"| {name} | " + " | ".join(f"{row[g][0]} ({row[g][1]}%)" for g in GAMES) + " |")
    by = defaultdict(lambda: defaultdict(list))
    for r in res:
        if "eval" in r:
            by[r["agent"]][r["game"]].append((r["eval"]["mean_score"], 100 * r["eval"]["win_rate"]))
    for agent, games in by.items():
        lines.append(f"| {agent} (2026) | " + " | ".join(cell(games[g]) if games[g] else "–" for g in GAMES) + " |")

    lines += ["", "## Aliens learning: win % per 500 training games", "",
              "| Run | 1–500 | 501–1000 | 1001–1500 | 1501–2000 |", "|---|---|---|---|---|"]  # fmt: skip
    for name, blocks in curves_2018().items():
        lines.append(f"| {name} | " + " | ".join(f"{b:.0f}%" for b in blocks) + " |")
    for r in res:
        if r.get("agent") == "ddqn2018" and r["game"] == "aliens":
            rows = read_csv(HERE / "runs" / r["name"] / "episodes.csv")
            blocks = win_blocks([int(x["win"]) for x in rows])
            lines.append(f"| {r['name']} | " + " | ".join(f"{b:.0f}%" for b in blocks) + " |")

    lines += ["", "## Training time", "", "| Run | Game ticks | Wall clock | Ticks/s |", "|---|---|---|---|",
              "| 2018 plain DQN, GTX 980 (report) | ~1.0M | ~12 h | ~23 |"]  # fmt: skip
    for r in res:
        if "train_wall_s" in r:
            t, s = r["train_ticks"], r["train_wall_s"]
            lines.append(f"| {r['name']} | {t / 1e6:.2f}M | {s / 60:.0f} min | {t / s:,.0f} |")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
