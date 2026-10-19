"""The 2018 agent's input, decision by decision, as one sprite sheet.

Replays a game with tools/record.py's play_2018 (CPU) and writes a PNG with one row per decision: the four
greyscale half-size frames the network saw, side by side, oldest first. With --check it also asserts that the
decisions (tick, action, Q-values) are the ones in a recording's JSON, so the sheet lines up with that clip.

    uv run python -m tools.stacks --out <blog>/public/blog/gvgai --name aliens-2018-stacks \
        --check <blog>/public/blog/gvgai/aliens-2018-agent.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from tools.record import play_2018


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default="aliens-2018-stacks")
    ap.add_argument("--game", default="aliens")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--weights", default="weights/aliens-ddqn2018-s0-agent.pt")
    ap.add_argument("--check", type=Path, help="a recording's JSON whose decisions this replay must match")
    a = ap.parse_args()

    rec = play_2018(a.weights, a.game, a.level, a.seed)
    stacks = rec["stacks"]
    if a.check:
        want = [d for d in json.loads(a.check.read_text())["decisions"] if d.get("q")]
        got = rec["decisions"]
        if len(want) != len(got) or any(
            (w["tick"], w["action"], w["q"]) != (g["tick"], g["action"], g["q"]) for w, g in zip(want, got)
        ):
            raise SystemExit(f"replay differs from {a.check}: {len(got)} decisions vs {len(want)}")
    n, frames, h, w = len(stacks), *stacks[0].shape
    sheet = np.zeros((n * h, frames * w), np.uint8)
    for k, stack in enumerate(stacks):
        for i in range(frames):
            sheet[k * h : (k + 1) * h, i * w : (i + 1) * w] = stack[i]
    a.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(sheet, "L").save(a.out / f"{a.name}.png", optimize=True)
    print(f"{a.name}.png: {n} decisions x {frames} frames of {w}x{h}, checked={bool(a.check)}")


if __name__ == "__main__":
    main()
