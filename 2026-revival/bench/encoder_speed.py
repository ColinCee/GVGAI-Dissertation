"""Milliseconds per training step (forward + backward + Adam) of an encoder alone, for layout and precision choices.

uv run python bench/encoder_speed.py
"""

import sys
import time
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.modern import STACK, impala_encoder  # noqa: E402

torch.backends.cudnn.benchmark = True


def nature() -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(STACK, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                         nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten())  # fmt: skip


def bench(name: str, enc: nn.Module, batch: int, amp: bool, channels_last: bool, iters: int = 100) -> None:
    enc = enc.cuda()
    x = torch.rand(batch, STACK, 55, 150, device="cuda")
    if channels_last:
        enc, x = enc.to(memory_format=torch.channels_last), x.contiguous(memory_format=torch.channels_last)
    opt = torch.optim.Adam(enc.parameters(), fused=True)

    def step() -> None:
        with torch.autocast("cuda", torch.bfloat16, enabled=amp):
            y = enc(x).float().sum()
        y.backward()
        opt.step()
        opt.zero_grad(set_to_none=False)

    for _ in range(20):
        step()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    print(f"{name} batch {batch} amp={amp} channels_last={channels_last}: "
          f"{(time.perf_counter() - t) / iters * 1000:.2f} ms", flush=True)  # fmt: skip


if __name__ == "__main__":
    bench("nature", nature(), 32, True, True)
    for batch in (32, 128):
        for amp in (True, False):
            for cl in (True, False):
                bench("impala", impala_encoder(), batch, amp, cl)
