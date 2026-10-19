"""Random play on Deceptizelda for many games, the case that once filled the server's heap (IMPROVEMENTS row 29).

    uv run python bench/deceptizelda_heap.py 5000

While it runs, `jcmd <java pid> GC.heap_info` shows the server's heap; it should stay flat.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgai.env import GvgaiEnv  # noqa: E402

if __name__ == "__main__":
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    env = GvgaiEnv("deceptizelda", levels=(0, 2), engine_seed=1, gray_half=True)
    print("java pid", env.comm.java.pid, flush=True)
    rng = random.Random(0)
    for game in range(games):
        env.reset()
        while True:
            *_, term, trunc, _ = env.step(rng.randrange(env.n_actions), 4)
            if term or trunc:
                break
        if game % 500 == 0:
            print("game", game, flush=True)
    env.close()
