"""The 2018 dissertation's final agent (DDQN + PER + loss penalty), ported to PyTorch.

Source: tag 2018-ddqn-per, src/QLearner/{Agent,Brain,Network,Replay,SumTree,State}.py. Every setting
and quirk is kept on purpose so this replays 2018, including ones that look like bugs:

- pixels go into the network as raw 0-255 values, no scaling;
- the reward stored with a transition is the score change caused by the previous action
  (computed when the next action is chosen), and at game end it is replaced by +/-1;
- on a loss every transition of the episode gets -0.5, clipped at -1;
- training only at episode end: int(2 * stacks / 32) minibatches;
- PER without importance weights; sampling uses an integer point in [0, int(total)];
- new transitions get their current TD error as priority (warmup: (|r| + 1e-4) ** 0.6);
- MSE loss averaged over actions, Adam(lr 2.5e-4, Keras decay 1e-4, eps 1e-7), global clipnorm 1;
- target network starts from its own random weights and is synced every 5000 stacks, checked at
  episode end; evaluation uses the weights from the last sync;
- epsilon 1.0 through 20k warmup stacks, then -0.0009 per episode down to 0.1.

`per_sample=True` runs the network one sample at a time as 2018 did (for the timing comparison);
the default batches the same arithmetic.
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from PIL import Image
from torch import nn

FRAMES_PER_STACK = 4
DOWNSCALE = 2


@dataclass
class Config:
    gamma: float = 0.99
    lr: float = 0.00025
    lr_decay: float = 1e-4
    batch_size: int = 32
    memory_size: int = 100_000
    warmup_stacks: int = 20_000
    target_update_stacks: int = 5_000
    eps_min: float = 0.1
    eps_episodes: int = 1000
    per_alpha: float = 0.6
    per_eps: float = 1e-4
    loss_penalty: float = 0.5
    per_sample: bool = False
    seed: int = 0
    extra: dict = field(default_factory=dict)


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """2018's State.resize_image: greyscale, then halve with Lanczos."""
    img = Image.fromarray(rgb).convert("L")
    return np.asarray(img.resize((img.width // DOWNSCALE, img.height // DOWNSCALE), Image.LANCZOS), np.uint8)


class QNet(nn.Module):
    """2018's Network.deep_q_network, with Keras's default init (Glorot uniform, zero bias)."""

    def __init__(self, h: int, w: int, n_actions: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(FRAMES_PER_STACK, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )  # fmt: skip
        with torch.no_grad():
            n = self.conv(torch.zeros(1, FRAMES_PER_STACK, h, w)).shape[1]
        self.head = nn.Sequential(nn.Linear(n, 512), nn.ReLU(), nn.Linear(512, n_actions))
        for m in self.modules():
            if isinstance(m, nn.Conv2d | nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x.float()))  # raw 0-255, as in 2018


class SumTree:
    """2018's SumTree, as arrays. Leaves hold priorities; data index = leaf - capacity + 1."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.write = 0
        self.filled = 0

    def total(self) -> float:
        return float(self.tree[0])

    def update(self, leaf: int, p: float) -> None:
        change = p - self.tree[leaf]
        self.tree[leaf] = p
        while leaf != 0:
            leaf = (leaf - 1) // 2
            self.tree[leaf] += change

    def add(self, p: float) -> int:
        slot = self.write
        self.update(slot + self.capacity - 1, p)
        self.write = (self.write + 1) % self.capacity
        self.filled = min(self.filled + 1, self.capacity)
        return slot

    def get(self, s: float) -> int:
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                return idx
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1


class Replay:
    """Each stack stored once as uint8.

    Episodes are added whole and in order, so a transition's next state is the following slot's
    state, except for an episode's last transition, whose next state is kept in `terminal_next`.
    The ring overwrites oldest first, so a following slot is only replaced after the slot itself.
    """

    def __init__(self, cfg: Config, h: int, w: int, rng: random.Random):
        n = cfg.memory_size
        self.cfg, self.rng = cfg, rng
        self.tree = SumTree(n)
        self.state = np.zeros((n, FRAMES_PER_STACK, h, w), np.uint8)
        self.action = np.zeros(n, np.int64)
        self.reward = np.zeros(n, np.float32)
        self.done = np.zeros(n, np.float32)
        self.terminal_next: dict[int, np.ndarray] = {}

    def next_state(self, idx: np.ndarray) -> np.ndarray:
        n = self.cfg.memory_size
        return np.stack([self.terminal_next[i] if self.done[i] else self.state[(i + 1) % n] for i in idx])

    def priority(self, error: float) -> float:
        return (abs(error) + self.cfg.per_eps) ** self.cfg.per_alpha

    def add(self, p, s, a, r, s2, d) -> None:
        slot = self.tree.add(p)
        self.terminal_next.pop(slot, None)
        self.state[slot] = s
        self.action[slot], self.reward[slot], self.done[slot] = a, r, d
        if d:
            self.terminal_next[slot] = s2

    def sample(self, k: int) -> list[int]:
        leaves: list[int] = []
        while len(leaves) < k:
            leaf = self.tree.get(self.rng.randint(0, int(self.tree.total())))
            # 2018 redrew empty slots and duplicates
            if leaf - self.tree.capacity + 1 < self.tree.filled and leaf not in leaves:
                leaves.append(leaf)
        return leaves


class Agent2018:
    def __init__(self, n_actions: int, frame_shape: tuple[int, int], cfg: Config, device: str = "cpu"):
        self.cfg, self.n_actions, self.device = cfg, n_actions, device
        self.rng = random.Random(cfg.seed)
        self.np_rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        h, w = frame_shape
        self.q = QNet(h, w, n_actions).to(device)
        self.target = QNet(h, w, n_actions).to(device)  # own random init, as 2018
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr, eps=1e-7)
        self.updates = 0
        self.replay = Replay(cfg, h, w, self.rng)
        self.epsilon = 1.0
        self.total_stacks = 0
        self.stacks_since_sync = 0
        self.synced_state: dict | None = None  # weights at last target sync, used for evaluation

    # --- acting -------------------------------------------------------------------------------------
    @torch.no_grad()
    def act(self, stack: np.ndarray, greedy: bool = False, net: nn.Module | None = None) -> int:
        if not greedy and self.np_rng.random() <= self.epsilon:
            return self.rng.randrange(self.n_actions)
        q = (net or self.q)(torch.from_numpy(stack[None]).to(self.device))
        return int(q.argmax(1).item())

    # --- learning -----------------------------------------------------------------------------------
    def _q(self, net: nn.Module, x: np.ndarray) -> torch.Tensor:
        t = torch.from_numpy(x).to(self.device)
        return torch.cat([net(s[None]) for s in t]) if self.cfg.per_sample else net(t)

    @torch.no_grad()
    def _targets(self, r: np.ndarray, s2: np.ndarray, d: np.ndarray) -> np.ndarray:
        q1, q2 = self._q(self.q, s2), self._q(self.target, s2)
        best = q2.gather(1, q1.argmax(1, keepdim=True)).squeeze(1).cpu().numpy()
        return np.where(d > 0, r, r + self.cfg.gamma * best).astype(np.float32)

    def remember_episode(self, transitions: list, warmup: bool) -> None:
        """2018's store_transition: penalise a lost episode, then add with PER priorities."""
        if not transitions:
            return
        if transitions[-1][2] == -1:  # terminal reward -1 means the game was lost
            transitions = [
                (s, a, max(r - self.cfg.loss_penalty, -1.0), s2, d) for s, a, r, s2, d in transitions
            ]
        rp = self.replay
        s, a, r, s2, d = (np.array(x) for x in zip(*transitions, strict=True))
        r, d = r.astype(np.float32), d.astype(np.float32)
        if warmup:
            err = r  # no network yet: 2018 used get_priority(reward, 0)
        else:
            with torch.no_grad():
                q = self._q(self.q, s).gather(1, torch.from_numpy(a)[:, None].to(self.device)).squeeze(1)
            err = q.cpu().numpy() - self._targets(r, s2, d)
        for i in range(len(transitions)):
            rp.add(rp.priority(float(err[i])), s[i], a[i], r[i], s2[i], d[i])

    def train_step(self) -> None:
        rp = self.replay
        leaves = rp.sample(self.cfg.batch_size)
        idx = np.array(leaves) - rp.tree.capacity + 1
        target = torch.from_numpy(self._targets(rp.reward[idx], rp.next_state(idx), rp.done[idx])).to(
            self.device
        )
        a = torch.from_numpy(rp.action[idx]).to(self.device)
        q = self._q(self.q, rp.state[idx])
        q_taken = q.gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            for leaf, e in zip(leaves, (q_taken - target).cpu().numpy(), strict=True):
                rp.tree.update(leaf, rp.priority(float(e)))
        # Keras fit on (s, q with the taken action replaced by target): MSE averaged over all actions.
        y = q.detach().clone()
        y[torch.arange(len(idx)), a] = target
        loss = ((q - y) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 1.0)
        for g in self.opt.param_groups:
            g["lr"] = self.cfg.lr / (1 + self.cfg.lr_decay * self.updates)  # Keras 2 `decay`
        self.opt.step()
        self.updates += 1

    def end_episode(self, transitions: list, decisions: int, warmup: bool) -> int:
        """2018's result(): remember, train, decay epsilon, maybe sync. Returns minibatches trained."""
        self.remember_episode(transitions, warmup)
        if warmup:
            return 0
        n = int(2 * decisions / self.cfg.batch_size)
        for _ in range(n):
            self.train_step()
        if self.epsilon > self.cfg.eps_min:
            self.epsilon -= (1 - self.cfg.eps_min) / self.cfg.eps_episodes
        if self.stacks_since_sync >= self.cfg.target_update_stacks:
            self.target.load_state_dict(self.q.state_dict())
            self.synced_state = {k: v.clone() for k, v in self.q.state_dict().items()}
            self.stacks_since_sync = 0
        return n


def play_episode(
    env, agent: Agent2018, train: bool = True, greedy: bool = False, net=None, level=None
) -> dict:
    """One game, following 2018's Agent.act/result tick by tick.

    `env` must be built with include_nil=True; agent action i is env action i + 1 (index 0 is NIL).
    """
    from collections import deque

    obs, info = env.reset(options=None if level is None else {"level": level})
    frames: deque = deque(maxlen=FRAMES_PER_STACK)
    step = 0  # 2018's current_episode_step: ticks seen from gameTick 2 on
    prev_state = prev_action = None
    prev_reward = 0.0
    prev_score = 0.0
    total_reward = 0.0
    transitions: list = []
    warmup = agent.total_stacks < agent.cfg.warmup_stacks
    decisions = 0
    score = 0.0
    while True:
        tick = env.comm.sso.gameTick
        if tick <= 1:  # 2018 ignored the first two frames
            action = -1
        else:
            frames.append(preprocess(obs))
            if step % FRAMES_PER_STACK == 0 and step > 0:
                stack = np.stack(frames)
                action = agent.act(stack, greedy=greedy, net=net)
                decisions += 1
                if prev_state is not None and train:
                    transitions.append((prev_state, prev_action, prev_reward, stack, 0.0))
                    agent.total_stacks += 1
                    agent.stacks_since_sync += 1
                prev_state, prev_action = stack, action
                # Score change since the last decision: credited to the action just chosen (2018 quirk).
                diff = score - prev_score
                prev_score = score
                prev_reward = float(np.sign(diff))
                total_reward += prev_reward
            else:
                action = -1 if prev_action is None else prev_action
            step += 1
        obs, _, term, trunc, info = env.step(action + 1)
        score = info["score"]
        if term or trunc:
            break
    frames.append(preprocess(obs))
    final = 1.0 if info["win"] else -1.0
    total_reward += final
    if train and prev_state is not None:
        transitions.append((prev_state, prev_action, final, np.stack(frames), 1.0))
        agent.stacks_since_sync += 1
    trained = agent.end_episode(transitions, step, warmup) if train else 0
    return {
        "score": score,
        "total_reward": total_reward,
        "steps": step + 1,
        "ticks": info["tick"],
        "win": int(info["win"]),
        "epsilon": agent.epsilon,
        "trained": trained,
        "warmup": int(warmup),
        "level": info["level"],
    }


def lr_now(agent: Agent2018) -> float:
    return agent.cfg.lr / (1 + agent.cfg.lr_decay * agent.updates)
