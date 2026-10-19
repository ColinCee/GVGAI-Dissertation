"""Can several seeds share the GPU better inside one process? IMPALA training step (forward + backward + Adam),
each seed captured as its own CUDA graph: replayed one after another on one stream, vs each on its own stream.

    uv run python bench/multiseed_speed.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.modern import STACK, impala_encoder  # noqa: E402

torch.backends.cudnn.benchmark = True


def captured_step(seed: int) -> torch.cuda.CUDAGraph:
    torch.manual_seed(seed)
    enc = impala_encoder().cuda().to(memory_format=torch.channels_last)
    x = torch.rand(32, STACK, 55, 150, device="cuda").contiguous(memory_format=torch.channels_last)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-4, fused=True, capturable=True)

    def step() -> None:
        opt.zero_grad(set_to_none=False)
        with torch.autocast("cuda", torch.bfloat16):
            y = enc(x).float().square().mean()
        y.backward()
        opt.step()

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    return g


def timed(fn, iters: int = 200) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


if __name__ == "__main__":
    for n in (1, 2, 3, 4):
        graphs = [captured_step(i) for i in range(n)]
        streams = [torch.cuda.Stream() for _ in range(n)]

        def one_stream(graphs=graphs) -> None:
            for g in graphs:
                g.replay()

        def many_streams(graphs=graphs, streams=streams) -> None:
            main = torch.cuda.current_stream()
            for g, s in zip(graphs, streams, strict=True):
                s.wait_stream(main)
                with torch.cuda.stream(s):
                    g.replay()
            for s in streams:
                main.wait_stream(s)

        a, b = timed(one_stream), timed(many_streams)
        print(f"{n} seeds: one stream {a:.2f} ms ({a / n:.2f} per seed), own streams {b:.2f} ms ({b / n:.2f} per seed)",
              flush=True)  # fmt: skip
