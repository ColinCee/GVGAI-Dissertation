"""Gymnasium wrapper around the GVGAI Java learning server (via GVGAI_GYM's engine).

Behaviour matches the 2018 learning-track setup the dissertation used:
- actions are the game's own available actions (no extra ACTION_NIL slot);
- a game lasts at most 2000 ticks and running out of time is a loss;
- each reset plays a random level from 0-2 unless one is fixed.
"""

import os
import random
import sys
from pathlib import Path

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
        self,
        game="aliens",
        levels=(0, 1, 2),
        max_ticks=2000,
        include_nil=False,
        version=0,
        frame_delay_ms=0,
        engine_seed=None,
        engine_dir=ENGINE_DIR,
        java_opts=(),
        raw_frames=True,
        light_obs=True,
        gray_half=False,
    ):
        # The engine sleeps frame_delay_ms per tick to pace the game for viewing (25 ms in 2018).
        # It has no effect on game logic.
        # engine_seed fixes the engine's per-level random seeds, so games replay exactly.
        build.ensure_built(Path(engine_dir))
        ClientCommGYM.JAVA = build.java()
        # Serial GC and a small heap: ~150 MB per server instead of 686 MB, same speed (RAM is shared);
        # 256 MB, not 128, so a long game's garbage collection can't stall a batched step.
        ClientCommGYM.JAVA_OPTS = ["-XX:+UseSerialGC", "-Xmx256m", f"-Dgvgai.longDelay={frame_delay_ms}"]
        if engine_seed is not None:
            ClientCommGYM.JAVA_OPTS.append(f"-Dgvgai.seed={engine_seed}")
        if raw_frames:  # pixels inline instead of PNG + frame file; the unmodified engine ignores this
            ClientCommGYM.JAVA_OPTS.append("-Dgvgai.rawFrames=true")
        if light_obs:  # skip the per-tick sprite copies only planning agents need
            ClientCommGYM.JAVA_OPTS.append("-Dgvgai.lightObs=true")
        if gray_half:  # frames come greyscale at half size (modern agents; 2018's replay uses PIL Lanczos)
            ClientCommGYM.JAVA_OPTS.append("-Dgvgai.grayHalf=true")
        ClientCommGYM.JAVA_OPTS += list(java_opts)
        self.game = game
        self.levels = tuple(levels)
        self.max_ticks = max_ticks
        self.include_nil = include_nil
        self._rng = random.Random()
        self.comm = ClientCommGYM.ClientCommGYM(game, version, self.levels[0], str(engine_dir))
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

    def step(self, action, repeat=1):
        """Play `action`; with repeat > 1 the server holds it that many ticks (stopping at the game's end
        or the tick limit) and returns only the last frame."""
        self.step_send(action, repeat)
        return self.step_recv()

    def step_send(self, action, repeat=1):
        """First half of step(): send the move without waiting (see gvgai/vec.py)."""
        assert not self._terminal, "call reset() after the game ends"
        self._before = self.comm.sso.gameTick
        repeat = max(1, min(repeat, self.max_ticks - self._before))
        self.comm.step_send(int(action) + self._offset, repeat)

    def step_recv(self):
        """Second half of step(): wait for the result."""
        before = self._before
        obs, reward, done, info = self.comm.step_recv()
        # the game-over message reports the tick the game ended on, one short of ticks played
        advanced = max(1, self.comm.sso.gameTick - before + (1 if done else 0))
        self._ticks += advanced
        # reset() already played one tick, so count the engine's own tick
        timeout = not done and self.comm.sso.gameTick >= self.max_ticks
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
            "advanced": advanced,
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
