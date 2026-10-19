"""GpuReplay must rebuild exactly the stacks, n-step returns and next states of the played episodes.

Frames are stamped with (game, decision index) so every sampled stack can be checked against what
was actually played: history padded with the episode's first frame, next state after an episode
end taken from its final frame.

    uv run python tests/replay_check.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.modern import STACK, GpuReplay, ModernConfig  # noqa: E402

N, H, W, STEPS = 3, 4, 5, 3000


def stamp(e: int, i: int) -> np.ndarray:
    f = np.zeros((H, W), np.uint8)
    f[0, 0], f[0, 1], f[0, 2] = e, i % 256, i // 256
    return f


def ident(f: torch.Tensor) -> tuple[int, int]:
    return int(f[0, 0]), int(f[0, 1]) + 256 * int(f[0, 2])


def main() -> int:
    rng = np.random.default_rng(0)
    bad = 0
    for n_step, per in [(1, False), (3, True)]:
        cfg = ModernConfig(n_step=n_step, per=per, buffer_decisions=N * 700)
        rp = GpuReplay(N, cfg.buffer_decisions, H, W, cfg, "cuda")
        # ground truth per game: list of episodes, each a list of (frame id, reward, term)
        frame_no = np.zeros(N, np.int64)
        episodes: list[list[list]] = [[[]] for _ in range(N)]
        finals: dict[tuple[int, int], int] = {}
        for _ in range(STEPS):
            obs = np.stack([stamp(e, frame_no[e]) for e in range(N)])
            r = rng.integers(-1, 2, N).astype(np.float32)
            ended = rng.random(N) < 0.03
            term = ended & (rng.random(N) < 0.5)
            trunc = ended & ~term
            final = np.stack([stamp(e, frame_no[e] + 1) for e in range(N)])
            rp.add(obs, np.zeros(N, np.int64), r, term, trunc, final)
            for e in range(N):
                episodes[e][-1].append((int(frame_no[e]), float(r[e]), bool(term[e]), bool(ended[e])))
                frame_no[e] += 1
                if ended[e]:
                    finals[(e, int(frame_no[e] - 1))] = int(frame_no[e])
                    frame_no[e] += 1
                    episodes[e].append([])
        lookup = {}
        for e in range(N):
            for ep in episodes[e]:
                for j, row in enumerate(ep):
                    lookup[(e, row[0])] = (ep, j)
        s, a, ret, s2, done, gn, w, flat = rp.sample(4096, torch.tensor(0.4, device="cuda"))
        for i in range(len(flat)):
            if float(w[i]) == 0:  # plain replay's rare unusable draw: weighted out of the loss
                continue
            e, fid = ident(s[i, -1])
            ep, j = lookup[(e, fid)]
            want_hist = [ep[max(0, j - k)][0] for k in range(STACK - 1, -1, -1)]
            got_hist = [ident(s[i, k])[1] for k in range(STACK)]
            # n-step return and next state
            want_ret, want_done, last = 0.0, False, j
            for d in range(n_step):
                row = ep[j + d]
                want_ret += cfg.gamma**d * row[1]
                last = j + d
                if row[3]:
                    want_done = row[2]
                    break
            if ep[last][3]:
                nxt = finals[(e, ep[last][0])]
                want_next = [
                    ep[max(0, last + 1 - k)][0] if last + 1 - k <= last else nxt for k in range(STACK - 1, -1, -1)
                ]
            else:
                want_next = [ep[max(0, last + 1 - k)][0] for k in range(STACK - 1, -1, -1)]
            got_next = [ident(s2[i, k])[1] for k in range(STACK)]
            ok = (got_hist == want_hist and got_next == want_next and abs(float(ret[i]) - want_ret) < 1e-5
                  and bool(done[i]) == want_done)  # fmt: skip
            if not ok:
                bad += 1
                if bad < 5:
                    print("mismatch", n_step, per, e, fid, got_hist, want_hist, got_next, want_next,
                          float(ret[i]), want_ret, bool(done[i]), want_done)  # fmt: skip
        print(f"n_step={n_step} per={per}: {len(flat)} samples, {bad} wrong")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
