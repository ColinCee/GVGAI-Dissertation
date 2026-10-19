"""Where the 2018 agent looks: a recording with a saliency map drawn over the screen.

For each decision, gradient × input of the chosen move's value says which pixels on screen would change that
value most (the gradient alone is a diffuse fog on this network). Kept per input frame, blurred a little and scaled by
the game's 99th percentile, it is drawn in tomato over the frame it belongs to.
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
    return (x.grad[0] * x[0]).abs().detach().numpy()  # gradient × input, per input frame: (4, h, w)


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
    share = np.stack([m.sum((1, 2)) / m.sum() for m in maps]).mean(0)
    print("attribution by input frame, oldest first:", np.round(share, 3))
    top = float(np.percentile(np.stack(maps), 99))
    fh, fw = frames[0].shape[:2]

    def heat(m: np.ndarray) -> np.ndarray:
        img = Image.fromarray(np.clip(m / top * 255, 0, 255).astype(np.uint8))
        img = img.resize((fw, fh), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(2))
        return np.asarray(img, np.float32)[..., None] / 255

    # Each video frame gets the heat of its own pixels, from the decision whose 4-frame input included it
    # (the stack is the frames at ticks t-3..t for a decision on tick t), so nothing is drawn where an
    # object used to be.
    owner = {d["tick"] - 3 + c: (k, c) for k, d in enumerate(decs) for c in range(4)}
    out_frames = []
    for t, f in enumerate(frames):
        if t not in owner:
            out_frames.append(f)
            continue
        k, c = owner[t]
        alpha = np.clip(heat(maps[k][c]) * 0.85, 0, 0.85)
        out_frames.append((f * (1 - alpha) + TOMATO * alpha).astype(np.uint8))

    encode(out_frames, out / f"{a.name}.mp4", a.fps, ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28",
                                                       "-movflags", "+faststart"])  # fmt: skip
    encode(out_frames, out / f"{a.name}.webm", a.fps, ["-c:v", "libvpx-vp9", "-crf", "40", "-b:v", "0"])
    mid = len(out_frames) // 2
    Image.fromarray(out_frames[mid].repeat(2, 0).repeat(2, 1)).save(out / f"{a.name}.png", optimize=True)
    info = {"game": a.game, "level": a.level, "seed": a.seed, "weights": a.weights, "win": rec["win"],
            "score": rec["score"], "ticks": rec["ticks"], "fps": a.fps, "scale": "99th percentile of all maps",
            "method": "|input × dQ/dinput| per input frame, drawn on that frame; blur 2 px"}  # fmt: skip
    (out / f"{a.name}.json").write_text(json.dumps(info, indent=1))
    print(info)


if __name__ == "__main__":
    main()
