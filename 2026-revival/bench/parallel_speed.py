"""Env throughput with N games running in parallel processes (random play, 4-tick action repeat)."""

import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path


def worker(game, seconds, q):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gvgai.env import GvgaiEnv

    env = GvgaiEnv(game)
    ticks = 0
    env.reset()
    t_end = time.perf_counter() + seconds
    t = 0
    while time.perf_counter() < t_end:
        if t % 4 == 0:
            act = random.randrange(env.n_actions)
        t += 1
        *_, term, trunc, _ = env.step(act)
        ticks += 1
        if term or trunc:
            env.reset()
            t = 0
    env.close()
    q.put(ticks)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="aliens")
    p.add_argument("--seconds", type=float, default=45)
    p.add_argument("--n", type=int, nargs="+", default=[1, 4, 8, 16])
    a = p.parse_args()
    for n in a.n:
        q = mp.Queue()
        procs = [mp.Process(target=worker, args=(a.game, a.seconds, q)) for _ in range(n)]
        for pr in procs:
            pr.start()
        total = sum(q.get() for _ in procs)
        for pr in procs:
            pr.join()
        print(
            f"{n:>2} games: {total / a.seconds:6.0f} ticks/s total, {total / a.seconds / n:4.0f} per game",
            flush=True,
        )
