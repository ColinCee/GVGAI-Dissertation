"""Many games at once for the modern agents: one game server per worker process.

Observations are already greyscale and halved (2018's preprocessing), so little data crosses
processes. The 2000-tick limit is reported as truncation here, so learners don't treat it as a
real end of game; evaluation still counts it as a loss (info["win"] is False).
"""

import gymnasium as gym
import numpy as np
from PIL import Image

from gvgai.env import GvgaiEnv


class Preprocessed(gym.Wrapper):
    """Greyscale, halved; each action held for `repeat` ticks, only the last frame kept."""

    def __init__(self, env: GvgaiEnv, repeat: int = 4):
        super().__init__(env)
        self.repeat = repeat
        h, w, _ = env.observation_space.shape
        self.observation_space = gym.spaces.Box(0, 255, (h // 2, w // 2), np.uint8)

    @staticmethod
    def _p(rgb: np.ndarray) -> np.ndarray:
        img = Image.fromarray(rgb).convert("L")
        return np.asarray(img.resize((img.width // 2, img.height // 2), Image.LANCZOS), np.uint8)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        return self._p(obs), info

    def step(self, action):
        # the server holds the action for `repeat` ticks (tests/repeat_check.py: same as repeating here)
        obs, total, term, trunc, info = self.env.step(action, self.repeat)
        ticks = info["advanced"]
        if info["timeout"]:
            term, trunc = False, True
        out = {"score": info["score"], "win": info["win"], "level": info["level"], "ticks": ticks}
        return self._p(obs), total, term, trunc, out


class InProcVec:
    """The subset of gymnasium's vector API the trainers use (SAME_STEP autoreset), in this process.

    Sends every game its move, then reads all the results: the Java servers play in parallel and
    there are no worker processes or pipes. Frames arrive greyscale at half size from the server
    (gray_half), so there is no image work here either. AsyncVectorEnv + PIL Lanczos is kept as
    `vector(..., processes=True)` for comparison.
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


def make(game: str, seed: int, levels=(0, 1, 2), repeat: int = 4):
    def thunk():
        return Preprocessed(GvgaiEnv(game, levels=levels, engine_seed=seed), repeat)

    return thunk


def vector(game: str, n: int, seed: int, levels=(0, 1, 2), repeat: int = 4, processes: bool = False):
    if not processes:
        return InProcVec(game, n, seed, levels, repeat)
    return gym.vector.AsyncVectorEnv(
        [make(game, seed * 1000 + i, levels, repeat) for i in range(n)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        context="spawn",
    )
