"""The 2018 agent's whole network over a stretch of a recorded game, for the blog's 3D explorer.

For each decision in the window: the 4 input frames, every map of the three convolution layers (after ReLU),
the 512 hidden units, the three move values, and for each move a saliency map (|input × gradient| per input
frame). Each layer is one PNG atlas: one tile per decision, stacked top to bottom; inside a tile the maps sit
in a grid with a 1 px gap. Each map (channel) is scaled to its own min..max over the window, so brightness is
comparable from frame to frame; saliency is scaled by its 99th percentile over the window and kept to 8
levels. CPU only.

    uv run python -m tools.explode --weights weights/aliens-ddqn2018-s0-agent.pt --out <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools.record import load_2018, play_2018

GAP = 24  # grey of the 1 px gaps between maps


def atlas(x: np.ndarray, cols: int, path: Path) -> dict:
    """x: (decisions, maps, h, w) in 0..255. Writes the atlas and returns its layout."""
    t, n, h, w = x.shape
    rows = -(-n // cols)
    th, tw = rows * (h + 1) + 1, cols * (w + 1) + 1
    img = np.full((t * th, tw), GAP, np.uint8)
    for d in range(t):
        for m in range(n):
            r, c = divmod(m, cols)
            y, z = d * th + 1 + r * (h + 1), 1 + c * (w + 1)
            img[y : y + h, z : z + w] = x[d, m]
    Image.fromarray(img).save(path, optimize=True)
    return {"png": path.name, "maps": n, "cols": cols, "rows": rows, "h": h, "w": w, "tile_h": th, "tile_w": tw}


def per_channel(x: np.ndarray) -> np.ndarray:
    lo = x.min(axis=(0, 2, 3), keepdims=True)
    hi = x.max(axis=(0, 2, 3), keepdims=True)
    return ((x - lo) / (hi - lo + 1e-8) * 255).round().astype(np.uint8)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--game", default="aliens")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--start", type=int, default=50, help="first decision of the window")
    p.add_argument("--count", type=int, default=30, help="decisions in the window")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rec = play_2018(a.weights, a.game, a.level, a.seed)
    decs = rec["decisions"][a.start : a.start + a.count]
    stacks = np.stack(rec["stacks"][a.start : a.start + a.count])
    names = rec["names"]
    h, w = stacks.shape[2:]
    net = load_2018(a.weights, h, w, len(names))
    relus = [m for m in net.conv if isinstance(m, torch.nn.ReLU)] + [net.head[1]]
    acts: list[list[np.ndarray]] = [[] for _ in relus]
    hooks = [
        m.register_forward_hook(lambda _m, _i, o, k=k: acts[k].append(o[0].detach().numpy()))
        for k, m in enumerate(relus)
    ]
    qs, sal = [], []
    for s in stacks:
        x = torch.from_numpy(s[None]).float().requires_grad_(True)
        q = net(x)[0]
        qs.append([round(float(v), 3) for v in q.detach()])
        row = []
        for i in range(len(names)):
            (g,) = torch.autograd.grad(q[i], x, retain_graph=True)
            row.append((g[0] * x[0]).abs().detach().numpy())
        sal.append(row)
    for hk in hooks:
        hk.remove()

    layers = [{"name": "input", **atlas(stacks, 2, out / "input.png")}]
    convs = [{"kernel": 8, "stride": 4}, {"kernel": 4, "stride": 2}, {"kernel": 3, "stride": 1}]
    for k, spec in enumerate(convs):
        x = np.stack(acts[k])
        grid = atlas(per_channel(x), (4, 8, 8)[k], out / f"conv{k + 1}.png")
        layers.append({"name": f"layer {k + 1}", **spec, **grid})
    dense = np.stack(acts[3])[:, :, None, None]  # (decisions, 512, 1, 1)
    layers.append({"name": "512 units", **atlas(per_channel(dense), 32, out / "dense.png")})

    s = np.array(sal)  # (decisions, moves, 4, h, w)
    top = float(np.percentile(s, 99))
    for i in range(len(names)):
        level = np.clip(s[:, i] / top * 8, 0, 7).astype(np.uint8)  # 8 levels: the panel only needs a heat map
        atlas((level * 36).astype(np.uint8), 2, out / f"saliency-{i}.png")

    info = {"game": a.game, "level": a.level, "seed": a.seed, "weights": a.weights, "frames": len(decs),
            "ticks": [d["tick"] for d in decs], "actions": names, "chosen": [d["action"] for d in decs],
            "q": qs, "layers": layers, "saliency": [f"saliency-{i}.png" for i in range(len(names))],
            "params": int(sum(p.numel() for p in net.parameters()))}  # fmt: skip
    (out / "net.json").write_text(json.dumps(info))
    print({k: v for k, v in info.items() if k not in ("q", "ticks", "chosen")})


if __name__ == "__main__":
    main()
