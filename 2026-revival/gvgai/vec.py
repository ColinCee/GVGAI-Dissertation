"""Many games at once for the modern agents, all driven from this process.

The 2000-tick limit is reported as truncation here, so learners don't treat it as a
real end of game; evaluation still counts it as a loss (info["win"] is False).
"""

import numpy as np

from gvgai.env import GvgaiEnv


class InProcVec:
    """The subset of gymnasium's vector API the trainers use (SAME_STEP autoreset), in this process.

    Sends every game its move, then reads all the results: the Java servers play in parallel and
    there are no worker processes or pipes. Frames arrive greyscale at half size from the server
    (gray_half), so there is no image work here either. (Worker processes with PIL resizing, the
    first version, gave 17,000 ticks/s for 8 games against 24,800: IMPROVEMENTS row 25.)
    """

    def __init__(self, game: str, n: int, seed: int, levels=(0, 1, 2), repeat: int = 4):
        self.envs = [GvgaiEnv(game, levels=levels, engine_seed=seed * 1000 + i, gray_half=True) for i in range(n)]
        self.repeat = repeat
        self.num_envs = n
        from concurrent.futures import ThreadPoolExecutor

        self.pool = ThreadPoolExecutor(n)
        self.single_action_space = self.envs[0].action_space
        self.single_observation_space = self.envs[0].observation_space

    def reset(self, seed=None):
        obs = [env.reset(seed=None if seed is None else seed + i)[0] for i, env in enumerate(self.envs)]
        return np.stack(obs), {}

    def step_async(self, actions) -> None:
        for env, a in zip(self.envs, actions, strict=True):
            env.step_send(int(a), self.repeat)

    def step_wait(self):
        n = self.num_envs
        obs, rew = [None] * n, np.zeros(n, np.float32)
        term, trunc = np.zeros(n, bool), np.zeros(n, bool)
        ticks = np.zeros(n, np.int64)
        finals: dict[int, tuple] = {}
        for e, env in enumerate(self.envs):
            o, r, t, _, info = env.step_recv()
            ticks[e] = info["advanced"]
            if info["timeout"]:
                t, tr = False, True
            else:
                tr = False
            rew[e], term[e], trunc[e] = r, t, tr
            if t or tr:
                finals[e] = (o, info)
            obs[e] = o
        if finals:
            # Resets in parallel: each waits on its server loading a level. One after another, a game
            # whose episodes last ~15 ticks (Deceptizelda, once learned) ran ~10x slower.
            for e, (o, _) in zip(finals, self.pool.map(lambda i: self.envs[i].reset(), list(finals)), strict=True):
                obs[e] = o
        out: dict = {"ticks": ticks}
        if finals:
            out["final_obs"] = np.stack([finals[e][0] if e in finals else obs[e] for e in range(n)])
            out["final_info"] = {k: np.array([finals[e][1][k] if e in finals else 0 for e in range(n)])
                                 for k in ("score", "win", "level")}  # fmt: skip
        return np.stack(obs), rew, term, trunc, out

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self) -> None:
        self.pool.shutdown()
        for env in self.envs:
            env.close()


def vector(game: str, n: int, seed: int, levels=(0, 1, 2), repeat: int = 4) -> InProcVec:
    return InProcVec(game, n, seed, levels, repeat)
