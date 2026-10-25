"""Trivial baselines: one action held for the whole game, and uniform random play, on the agents' evaluation.

    uv run python -m baselines.trivial            # writes results/baseline-<game>-<policy>.json
    uv run python -m baselines.trivial --game aliens --policy hold-use --games 100 --seeds 0-4
                                                  # baseline-aliens-hold-use-e100-s<k>.json, one per seed

By default, the same levels, seed (7) and 50 games as a seed-0 agent's evaluation, so a learned agent that
matches a row exactly has learned that policy. With --seeds, seed k plays the test games of an agent run
with --seed k (engine seed k + 7), so a baseline sits on exactly the games the agents are scored on.
"""

import argparse
import json

import numpy as np

from agents.evaluate import evaluate
from gvgai.games import GAMES, ROOT, game_levels


class Fixed:
    """What evaluate expects: .q with eval()/train(), .device, act(stacks, eps)."""

    device = "cpu"

    class q:
        @staticmethod
        def eval():
            pass

        @staticmethod
        def train():
            pass

    def __init__(self, action: int | None, n_actions: int):
        self.action, self.n, self.rng = action, n_actions, np.random.default_rng(0)

    def act(self, stacks, eps, actor=False):
        k = stacks.shape[0]
        return np.full(k, self.action) if self.action is not None else self.rng.integers(0, self.n, k)


def main() -> None:
    from gvgai.vec import vector

    p = argparse.ArgumentParser()
    p.add_argument("--game", default=None, help="one game (default: all)")
    p.add_argument("--policy", default=None, help="e.g. hold-use or random (default: all)")
    p.add_argument("--games", type=int, default=50)
    p.add_argument("--seeds", default=None, help="agent seeds to match, e.g. 0-4 (default: seed 0, one file)")
    a = p.parse_args()
    seeds = None
    if a.seeds:
        lo, _, hi = a.seeds.partition("-")
        seeds = range(int(lo), int(hi or lo) + 1)
    for g in [a.game] if a.game else GAMES:
        lv = game_levels(g)
        env = vector(g, 1, 0, lv, 4)
        names = list(env.envs[0].action_names)
        n = len(names)
        env.close()
        for action in [*range(n), None]:
            label = "random" if action is None else f"hold-{names[action].removeprefix('ACTION_').lower()}"
            if a.policy and label != a.policy:
                continue
            for k in seeds if seeds is not None else [None]:
                ev = evaluate(Fixed(action, n), g, 8, a.games, lv, 7 + (k or 0), 4)
                name = f"baseline-{g}-{label}" + (f"-e{a.games}-s{k}" if k is not None else "")
                res = {"name": name, "agent": f"baseline {label}", "game": g, "eval": ev}
                (ROOT / "results" / f"{name}.json").write_text(json.dumps(res, indent=1))
                print(f"{name}: {ev['mean_score']:.1f} ({ev['win_rate']:.0%})", flush=True)


if __name__ == "__main__":
    main()
