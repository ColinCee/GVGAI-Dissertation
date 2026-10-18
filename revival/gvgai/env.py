"""Gymnasium wrapper around the GVGAI Java learning server (via GVGAI_GYM's engine).

Behaviour matches the 2018 learning-track setup the dissertation used:
- actions are the game's own available actions (no extra ACTION_NIL slot);
- a game lasts at most 2000 ticks and running out of time is a loss;
- each reset plays a random level from 0-2 unless one is fixed.
"""

import os
import random
import sys

import gymnasium as gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE_DIR = os.path.join(HERE, "..", "engine")
sys.path.insert(0, os.path.join(HERE, "vendor"))
sys.path.insert(0, ENGINE_DIR)
import build  # noqa: E402
import ClientCommGYM  # noqa: E402


class GvgaiEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self, game="aliens", levels=(0, 1, 2), max_ticks=2000, include_nil=False, version=0, frame_delay_ms=0
    ):
        # The engine sleeps frame_delay_ms per tick to pace the game for viewing (25 ms in 2018).
        # It has no effect on game logic.
        build.ensure_built()
        ClientCommGYM.JAVA = build.java()
        ClientCommGYM.JAVA_OPTS = [f"-Dgvgai.longDelay={frame_delay_ms}"]
        self.game = game
        self.levels = tuple(levels)
        self.max_ticks = max_ticks
        self.include_nil = include_nil
        self._rng = random.Random()
        self.comm = ClientCommGYM.ClientCommGYM(game, version, self.levels[0], ENGINE_DIR)
        # The client prefixes ACTION_NIL; index i of our action space maps to client index i + offset.
        self.action_names = self.comm.actions() if include_nil else self.comm.actions()[1:]
        self._offset = 0 if include_nil else 1
        self.n_actions = len(self.action_names)
        self.action_space = gym.spaces.Discrete(self.n_actions)
        img = self.comm.sso.image
        self.observation_space = gym.spaces.Box(0, 255, shape=img.shape, dtype=np.uint8)
        self._ticks = 0
        self._terminal = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        level = (options or {}).get("level", self._rng.choice(self.levels))
        obs = self.comm.reset(level)
        self._ticks = 0
        self._terminal = False
        self.level = level
        return obs, {"level": level}

    def step(self, action):
        assert not self._terminal, "call reset() after the game ends"
        obs, reward, done, info = self.comm.step(int(action) + self._offset)
        self._ticks += 1
        timeout = not done and self._ticks >= self.max_ticks
        winner = info["winner"]
        if timeout:
            winner = "PLAYER_LOSES"
        self._terminal = done or timeout
        out = {
            "winner": winner,
            "win": winner == "PLAYER_WINS",
            "score": self.comm.sso.gameScore,
            "tick": self._ticks,
            "timeout": timeout,
            "level": self.level,
        }
        # The 2018 engine ended the game itself at 2000 ticks, so timeout is a real terminal state.
        return obs, float(reward), self._terminal, False, out

    def close(self):
        comm = self.comm
        comm.java.kill()
        comm.java.wait()
        comm.io.socket.close()
        comm.io.logfile.close()
        comm.tempDir.cleanup()
