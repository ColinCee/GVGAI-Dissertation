"""The 2018 agent's network over a stretch of a recorded game, for an exploded 3D view on the blog.

For each decision in the window: the newest greyscale input frame, the post-ReLU activations of the three
convolution layers and the three move values. Per layer, the 4 maps that vary most over the window are kept,
each scaled to its own min..max over the whole window so brightness is comparable from frame to frame.
Written as one PNG atlas per layer (one row per decision, maps side by side) plus a JSON. CPU only.

    uv run python -m tools.explode --weights weights/aliens-ddqn2018-s0-agent.pt --out <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools.record import load_2018, play_2018

MAPS = 4


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--game", default="aliens")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--start", type=int, default=40, help="first decision of the window")
    p.add_argument("--count", type=int, default=60, help="decisions in the window")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rec = play_2018(a.weights, a.game, a.level, a.seed)
    decs = rec["decisions"][a.start : a.start + a.count]
    stacks = rec["stacks"][a.start : a.start + a.count]
    names = rec["names"]
    h, w = stacks[0].shape[1:]
    net = load_2018(a.weights, h, w, len(names))
    convs = [m for m in net.conv if isinstance(m, torch.nn.ReLU)]
    acts: list[list[np.ndarray]] = [[] for _ in convs]
    hooks = [m.register_forward_hook(lambda _m, _i, o, k=k: acts[k].append(o[0].numpy())) for k, m in enumerate(convs)]
    qs = []
    with torch.no_grad():
        for s in stacks:
            qs.append(net(torch.from_numpy(s[None]))[0].tolist())
    for hk in hooks:
        hk.remove()

    layers = []
    inp = np.concatenate([s[-1] for s in stacks], 0)  # newest frame of each input, stacked by decision
    Image.fromarray(inp).save(out / "input.png", optimize=True)
    layers.append({"name": "input", "png": "input.png", "maps": 1, "h": h, "w": w, "channels": 4})
    for k, layer in enumerate(acts):
        x = np.stack(layer)  # (decisions, channels, h, w)
        pick = np.argsort(-x.std(axis=(0, 2, 3)))[:MAPS]
        rows = []
        for t in range(len(x)):
            maps = []
            for c in pick:
                m = x[:, c]
                lo, hi = m.min(), m.max()
                maps.append(((x[t, c] - lo) / (hi - lo + 1e-8) * 255).round().astype(np.uint8))
            rows.append(np.concatenate(maps, 1))
        png = f"conv{k + 1}.png"
        Image.fromarray(np.concatenate(rows, 0)).save(out / png, optimize=True)
        layers.append({"name": f"layer {k + 1}", "png": png, "maps": MAPS, "h": x.shape[2], "w": x.shape[3],
                       "channels": x.shape[1], "kept": [int(c) for c in pick]})  # fmt: skip
    info = {"game": a.game, "level": a.level, "seed": a.seed, "weights": a.weights, "frames": len(decs),
            "ticks": [d["tick"] for d in decs], "actions": names, "chosen": [d["action"] for d in decs],
            "q": [[round(v, 3) for v in q] for q in qs], "layers": layers}  # fmt: skip
    (out / "explode.json").write_text(json.dumps(info))
    print({k: v for k, v in info.items() if k not in ("q", "ticks", "chosen")})


if __name__ == "__main__":
    main()
