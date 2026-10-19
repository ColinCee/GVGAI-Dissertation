"""Where the 2018 agent looks: a recording with a saliency map drawn over the screen.

For each decision, gradient × input of the chosen move's value says which pixels on screen would change that
value most (the gradient alone is a diffuse fog on this network). Summed over the 4 frames, blurred a little and
scaled by the game's 99th percentile, it is drawn in tomato over the colour frame and held until the next decision.
CPU only.

    uv run python -m tools.saliency --weights weights/aliens-ddqn2018-s0-agent.pt --level 0 --seed 2 --out <dir>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

from tools.record import encode, load_2018, play_2018

TOMATO = np.array([233, 88, 52], np.float32)


def saliency(net: torch.nn.Module, stack: np.ndarray, action: int) -> np.ndarray:
    x = torch.from_numpy(stack[None]).float().requires_grad_(True)
    net(x)[0, action].backward()
    return (x.grad[0] * x[0]).abs().sum(0).detach().numpy()  # gradient × input: only pixels with something on them


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--game", default="aliens")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--fps", type=float, default=15)
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="aliens-2018-saliency")
    a = p.parse_args()
    out = Path(a.out)

    rec = play_2018(a.weights, a.game, a.level, a.seed)
    frames, decs, names = rec["frames"], rec["decisions"], rec["names"]
    h, w = rec["stacks"][0].shape[1:]
    net = load_2018(a.weights, h, w, len(names))
    maps = [saliency(net, s, names.index(d["action"])) for s, d in zip(rec["stacks"], decs, strict=True)]
    top = float(np.percentile(np.stack(maps), 99))
    fh, fw = frames[0].shape[:2]
    heat = []
    for m in maps:
        img = Image.fromarray(np.clip(m / top * 255, 0, 255).astype(np.uint8))
        img = img.resize((fw, fh), Image.Resampling.BILINEAR)
        heat.append(np.asarray(img.filter(ImageFilter.GaussianBlur(2)), np.float32)[..., None] / 255)

    out_frames, k = [], -1
    for t, f in enumerate(frames):
        while k + 1 < len(decs) and decs[k + 1]["tick"] < t:
            k += 1
        if k < 0:
            out_frames.append(f)
            continue
        alpha = np.clip(heat[k] * 0.85, 0, 0.85)
        out_frames.append((f * (1 - alpha) + TOMATO * alpha).astype(np.uint8))

    encode(out_frames, out / f"{a.name}.mp4", a.fps, ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28",
                                                       "-movflags", "+faststart"])  # fmt: skip
    encode(out_frames, out / f"{a.name}.webm", a.fps, ["-c:v", "libvpx-vp9", "-crf", "40", "-b:v", "0"])
    mid = len(out_frames) // 2
    Image.fromarray(out_frames[mid].repeat(2, 0).repeat(2, 1)).save(out / f"{a.name}.png", optimize=True)
    info = {"game": a.game, "level": a.level, "seed": a.seed, "weights": a.weights, "win": rec["win"],
            "score": rec["score"], "ticks": rec["ticks"], "fps": a.fps, "scale": "99th percentile of all maps",
            "method": "|input × d Q(chosen move) / d input| summed over the 4 frames, Gaussian blur 2 px"}  # fmt: skip
    (out / f"{a.name}.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
