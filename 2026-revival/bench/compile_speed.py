"""Does torch.compile speed up a training step? Encoder forward + backward + Adam, batch 32, bf16, channels-last:
eager inside a CUDA graph (what agents/modern.py does) vs compiled, with and without CUDA graphs.

    uv run python bench/compile_speed.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.modern import STACK, impala_encoder  # noqa: E402
from bench.encoder_speed import nature  # noqa: E402

torch.backends.cudnn.benchmark = True


def make(enc_fn, compile_mode: str | None):
    enc = enc_fn().cuda().to(memory_format=torch.channels_last)
    x = torch.rand(32, STACK, 55, 150, device="cuda").contiguous(memory_format=torch.channels_last)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-4, fused=True, capturable=True)
    f = torch.compile(enc, mode=compile_mode) if compile_mode else enc

    def step() -> None:
        opt.zero_grad(set_to_none=False)
        with torch.autocast("cuda", torch.bfloat16):
            y = f(x).float().square().mean()
        y.backward()
        opt.step()

    return step


def graphed(step):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    return g.replay


def timed(fn, iters: int = 300) -> float:
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


if __name__ == "__main__":
    for name, enc in [("nature", nature), ("impala", impala_encoder)]:
        base = timed(graphed(make(enc, None)))
        fused = timed(graphed(make(enc, "max-autotune-no-cudagraphs")))
        print(f"{name}: eager in a CUDA graph {base:.2f} ms, compiled in a CUDA graph {fused:.2f} ms "
              f"({base / fused:.2f}x)", flush=True)  # fmt: skip
