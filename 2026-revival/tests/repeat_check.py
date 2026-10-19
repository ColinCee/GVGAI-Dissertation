"""The server-side action repeat must match repeating the action from Python, decision by decision.

uv run python tests/repeat_check.py
"""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgai.env import GvgaiEnv  # noqa: E402
from gvgai.games import GAMES  # noqa: E402

REPEAT, DECISIONS, SEED = 4, 400, 7


def main() -> int:
    bad = 0
    for game in GAMES:
        py = GvgaiEnv(game, levels=(0, 1, 2), engine_seed=SEED)
        jv = GvgaiEnv(game, levels=(0, 1, 2), engine_seed=SEED)
        moves = random.Random(game)
        try:
            for level in (0, 1, 2):
                py.reset(options={"level": level})
                jv.reset(options={"level": level})
                for d in range(DECISIONS):
                    a = moves.randrange(py.n_actions)
                    for _ in range(REPEAT):
                        o1, r1, t1, tr1, i1 = py.step(a)
                        if t1 or tr1:
                            break
                    o2, r2, t2, tr2, i2 = jv.step(a, REPEAT)
                    same = np.array_equal(o1, o2) and i1["score"] == i2["score"] and (t1, tr1) == (t2, tr2)
                    if not same or i1["tick"] != i2["tick"]:
                        print(f"{game} level {level}: differs at decision {d} (ticks {i1['tick']} vs {i2['tick']})")
                        bad += 1
                        break
                    if t1 or tr1:
                        break
            print(f"{game}: {'ok' if not bad else 'DIFFERS'}")
        finally:
            py.close()
            jv.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
