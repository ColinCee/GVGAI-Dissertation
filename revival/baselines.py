"""Trivial baselines: one action held for the whole game, and uniform random play, on the agents' evaluation.

    uv run python baselines.py            # writes results/baseline-<game>-<policy>.json

Same levels, seeds (7) and 50 games as a seed-0 agent's evaluation, so a learned agent that matches a row
exactly has learned that policy.
"""

import json

import numpy as np

from report import GAMES, HERE
from train2018 import DEFAULT_LEVELS
from train_modern import evaluate


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

    for g in GAMES:
        levels = DEFAULT_LEVELS.get(g, (0, 1, 2))
        env = vector(g, 1, 0, levels, 4)
        names = list(env.envs[0].action_names)
        n = len(names)
        env.close()
        for action in [*range(n), None]:
            ev = evaluate(Fixed(action, n), g, 8, 50, levels, 7, 4)
            label = "random" if action is None else f"hold-{names[action].removeprefix('ACTION_').lower()}"
            res = {"name": f"baseline-{g}-{label}", "agent": f"baseline {label}", "game": g, "eval": ev}
            (HERE / "results" / f"{res['name']}.json").write_text(json.dumps(res, indent=1))
            print(f"{g} {label}: {ev['mean_score']:.1f} ({ev['win_rate']:.0%})", flush=True)


if __name__ == "__main__":
    main()
