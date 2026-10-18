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
    def __init__(self, env: GvgaiEnv):
        super().__init__(env)
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
        obs, r, term, trunc, info = self.env.step(action)
        if info["timeout"]:
            term, trunc = False, True
        return self._p(obs), r, term, trunc, info


def make(game: str, seed: int, levels=(0, 1, 2)):
    def thunk():
        return Preprocessed(GvgaiEnv(game, levels=levels, engine_seed=seed))

    return thunk


def vector(game: str, n: int, seed: int, levels=(0, 1, 2)) -> gym.vector.AsyncVectorEnv:
    return gym.vector.AsyncVectorEnv(
        [make(game, seed * 1000 + i, levels) for i in range(n)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
        context="spawn",
    )
