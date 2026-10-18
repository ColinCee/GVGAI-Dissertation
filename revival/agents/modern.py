"""Modern value-based agent, from DQN-with-today's-defaults up to Rainbow, one switch per improvement.

The Rainbow parts (noisy nets, C51 projection, PER with importance weights) follow CleanRL's
rainbow_atari.py (MIT, https://github.com/vwxyzjn/cleanrl). Differences that matter here:
- many games at once (gvgai/vec.py); replay keeps one frame per decision per game and rebuilds
  stacks of 4 at sampling time, so 1M decisions take ~8 GB rather than ~33 GB;
- each decision repeats the action for `action_repeat` ticks (Atari's default skip of 4;
  2018 also acted every 4 ticks); budgets are counted in game ticks.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

STACK = 4


@dataclass
class ModernConfig:
    double: bool = True
    dueling: bool = False
    distributional: bool = False
    noisy: bool = False
    n_step: int = 1
    per: bool = False
    huber: bool = True
    lr: float = 1e-4
    adam_eps: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 32
    buffer_decisions: int = 1_000_000
    learning_starts: int = 20_000  # decisions
    replay_ratio: float = 0.25  # gradient updates per decision (CleanRL: train every 4 steps)
    target_every: int = 250  # updates; CleanRL DQN: every 1000 steps = 250 updates
    eps_start: float = 1.0
    eps_end: float = 0.01
    eps_fraction: float = 0.1  # of the decision budget
    per_alpha: float = 0.5
    per_beta: float = 0.4
    per_eps: float = 1e-6
    n_atoms: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    clip_reward: bool = True
    grad_clip: float = 10.0
    action_repeat: int = 4


def rainbow() -> ModernConfig:
    # CleanRL rainbow_atari defaults; target every 8000 steps = 2000 updates
    return ModernConfig(dueling=True, distributional=True, noisy=True, n_step=3, per=True, lr=6.25e-5,
                        adam_eps=1.5e-4, eps_start=0.0, eps_end=0.0, target_every=2000)  # fmt: skip


class NoisyLinear(nn.Module):
    def __init__(self, n_in: int, n_out: int, std_init: float = 0.5):
        super().__init__()
        self.n_in, self.n_out = n_in, n_out
        self.weight_mu = nn.Parameter(torch.empty(n_out, n_in))
        self.weight_sigma = nn.Parameter(torch.empty(n_out, n_in))
        self.bias_mu = nn.Parameter(torch.empty(n_out))
        self.bias_sigma = nn.Parameter(torch.empty(n_out))
        self.register_buffer("weight_eps", torch.empty(n_out, n_in))
        self.register_buffer("bias_eps", torch.empty(n_out))
        r = 1 / math.sqrt(n_in)
        self.weight_mu.data.uniform_(-r, r)
        self.bias_mu.data.uniform_(-r, r)
        self.weight_sigma.data.fill_(std_init / math.sqrt(n_in))
        self.bias_sigma.data.fill_(std_init / math.sqrt(n_out))
        self.reset_noise()

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return x.sign() * x.abs().sqrt()

    def reset_noise(self) -> None:
        # factorised Gaussian noise
        e_in = self._f(torch.randn(self.n_in, device=self.weight_mu.device))
        e_out = self._f(torch.randn(self.n_out, device=self.weight_mu.device))
        self.weight_eps.copy_(e_out.outer(e_in))
        self.bias_eps.copy_(e_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_eps,
                            self.bias_mu + self.bias_sigma * self.bias_eps)  # fmt: skip
        return F.linear(x, self.weight_mu, self.bias_mu)


class Net(nn.Module):
    def __init__(self, h: int, w: int, n_actions: int, cfg: ModernConfig):
        super().__init__()
        self.cfg, self.n_actions = cfg, n_actions
        self.atoms = cfg.n_atoms if cfg.distributional else 1
        self.conv = nn.Sequential(
            nn.Conv2d(STACK, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )  # fmt: skip
        with torch.no_grad():
            n = self.conv(torch.zeros(1, STACK, h, w)).shape[1]

        def lin(i: int, o: int) -> nn.Module:
            return NoisyLinear(i, o) if cfg.noisy else nn.Linear(i, o)

        self.adv = nn.Sequential(lin(n, 512), nn.ReLU(), lin(512, n_actions * self.atoms))
        self.val = nn.Sequential(lin(n, 512), nn.ReLU(), lin(512, self.atoms)) if cfg.dueling else None
        self.register_buffer("support", torch.linspace(cfg.v_min, cfg.v_max, cfg.n_atoms))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Q-values [B, A], or log-probabilities [B, A, atoms] when distributional."""
        h = self.conv(x.float() / 255.0)
        a = self.adv(h).view(-1, self.n_actions, self.atoms)
        if self.val is not None:
            a = self.val(h).view(-1, 1, self.atoms) + a - a.mean(1, keepdim=True)
        return F.log_softmax(a, 2) if self.cfg.distributional else a.squeeze(2)

    def q(self, x: torch.Tensor) -> torch.Tensor:
        out = self(x)
        return (out.exp() * self.support).sum(2) if self.cfg.distributional else out

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class SumTree:
    """Array sum tree with batched sampling and updates."""

    def __init__(self, capacity: int):
        self.size = 1
        while self.size < capacity:
            self.size *= 2
        self.tree = np.zeros(2 * self.size)
        self.max_p = 1.0

    def update(self, idx: np.ndarray, p: np.ndarray) -> None:
        i = np.asarray(idx) + self.size
        self.tree[i] = p
        i = np.unique(i // 2)
        while i[0] >= 1:
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]
            i = np.unique(i // 2)
            if i[0] == 0:
                break

    def total(self) -> float:
        return float(self.tree[1])

    def find(self, v: np.ndarray) -> np.ndarray:
        i = np.ones(len(v), np.int64)
        v = v.copy()
        while i[0] < self.size:
            left = self.tree[2 * i]
            go_right = v > left
            v = np.where(go_right, v - left, v)
            i = 2 * i + go_right
        return i - self.size


class FrameReplay:
    """One ring per game of single frames; a slot holds the frame a decision was made on.

    Slot k's next frame is slot k+1's, unless the episode ended at k (then `final` holds it).
    `start[k]` is the slot where k's episode began, for padding stacks at an episode start.
    """

    def __init__(self, n_envs: int, capacity: int, h: int, w: int, cfg: ModernConfig, rng: np.random.Generator):
        self.n, self.cap, self.cfg, self.rng = n_envs, capacity // n_envs, cfg, rng
        self.frames = np.zeros((n_envs, self.cap, h, w), np.uint8)
        self.action = np.zeros((n_envs, self.cap), np.int64)
        self.reward = np.zeros((n_envs, self.cap), np.float32)
        self.term = np.zeros((n_envs, self.cap), np.bool_)  # game over (no bootstrap)
        self.end = np.zeros((n_envs, self.cap), np.bool_)  # episode ended here (term or timeout)
        self.start = np.zeros((n_envs, self.cap), np.int64)  # absolute decision count of episode start
        self.final: dict[tuple[int, int], np.ndarray] = {}
        self.t = np.zeros(n_envs, np.int64)  # decisions written per env (absolute)
        self.ep_start = np.zeros(n_envs, np.int64)
        self.tree = SumTree(n_envs * self.cap) if cfg.per else None

    def add(self, obs: np.ndarray, action, reward, term, trunc, final_obs: np.ndarray) -> None:
        """obs[e]: newest frame the action was chosen on; final_obs[e] used when the episode ended."""
        for e in range(self.n):
            k = self.t[e] % self.cap
            self.final.pop((e, k), None)
            self.frames[e, k] = obs[e]
            # reward clipped per decision (Atari convention), so every preset sees the same scale
            r = np.sign(reward[e]) if self.cfg.clip_reward else reward[e]
            self.action[e, k], self.reward[e, k] = action[e], r
            self.term[e, k], self.end[e, k] = term[e], term[e] or trunc[e]
            self.start[e, k] = self.ep_start[e]
            if self.end[e, k]:
                self.final[(e, k)] = final_obs[e]
                self.ep_start[e] = self.t[e] + 1
            self.t[e] += 1
        if self.tree is not None:
            idx = np.arange(self.n) * self.cap + (self.t - 1) % self.cap
            self.tree.update(idx, np.full(self.n, self.tree.max_p**self.cfg.per_alpha))

    def _valid(self, e: np.ndarray, abs_t: np.ndarray) -> np.ndarray:
        # needs n_step following decisions written (or an episode end before them) and STACK-1 history
        newest = self.t[e] - 1
        # strictly before the newest decision: its next frame isn't written until the next step
        ok = (abs_t >= self.t[e] - self.cap + STACK) & (abs_t <= newest - self.cfg.n_step)
        return ok

    def _stack(self, e: np.ndarray, abs_t: np.ndarray) -> np.ndarray:
        offs = np.arange(-STACK + 1, 1)
        ts = abs_t[:, None] + offs[None, :]
        ts = np.maximum(ts, self.start[e, abs_t % self.cap][:, None])
        return self.frames[e[:, None], ts % self.cap]

    def sample(self, k: int, beta: float = 0.4):
        cfg = self.cfg
        while True:
            if self.tree is not None:
                total = self.tree.total()
                v = (np.arange(k) + self.rng.random(k)) * total / k
                flat = self.tree.find(v)
                e, slot = flat // self.cap, flat % self.cap
                # absolute decision index for that slot
                abs_t = self.t[e] - 1 - ((self.t[e] - 1 - slot) % self.cap)
            else:
                e = self.rng.integers(0, self.n, k)
                abs_t = self.t[e] - 1 - self.rng.integers(0, np.minimum(self.t[e], self.cap))
            ok = self._valid(e, abs_t)
            if ok.all():
                break
            if self.tree is None or ok.mean() > 0.5:
                e, abs_t = e[ok], abs_t[ok]
                if self.tree is None and len(e) < k:
                    continue
                break
        s = self._stack(e, abs_t)
        # n-step return, stopping at an episode end
        ret = np.zeros(len(e), np.float32)
        done = np.zeros(len(e), np.bool_)
        alive = np.ones(len(e), np.bool_)
        steps = np.zeros(len(e), np.int64)
        last = abs_t.copy()
        for j in range(cfg.n_step):
            tj = abs_t + j
            kj = tj % self.cap
            r = self.reward[e, kj]
            ret += np.where(alive, (cfg.gamma**j) * r, 0)
            steps += alive
            last = np.where(alive, tj, last)
            ended = self.end[e, kj] & alive
            done |= self.term[e, kj] & alive
            alive &= ~ended
        # next stack: after `last`; at an episode end it is the final frame's stack
        nxt_t = last + 1
        assert (nxt_t <= self.t[e] - 1).all()  # guaranteed by _valid
        s2 = self._stack(e, nxt_t)
        for i in range(len(e)):
            lk = last[i] % self.cap
            if self.end[e[i], lk]:
                hist = self._stack(e[i : i + 1], last[i : i + 1])[0]
                s2[i] = np.concatenate([hist[1:], self.final[(int(e[i]), int(lk))][None]])
        weights = np.ones(len(e), np.float32)
        flat = e * self.cap + abs_t % self.cap
        if self.tree is not None:
            p = self.tree.tree[flat + self.tree.size] / self.tree.total()
            filled = int(np.minimum(self.t, self.cap).sum())
            weights = (filled * p) ** (-beta)
            weights = (weights / weights.max()).astype(np.float32)
        a = self.action[e, abs_t % self.cap]
        return s, a, ret, s2, done, (cfg.gamma**steps).astype(np.float32), weights, flat

    def update_priorities(self, flat: np.ndarray, err: np.ndarray) -> None:
        if self.tree is None:
            return
        p = np.abs(err) + self.cfg.per_eps
        self.tree.max_p = max(self.tree.max_p, float(p.max()))
        self.tree.update(flat, p**self.cfg.per_alpha)


class ModernAgent:
    def __init__(self, n_envs: int, h: int, w: int, n_actions: int, cfg: ModernConfig, device: str, seed: int):
        self.cfg, self.device, self.n_actions = cfg, device, n_actions
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.q = Net(h, w, n_actions, cfg).to(device)
        self.target = Net(h, w, n_actions, cfg).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr, eps=cfg.adam_eps)
        self.replay = FrameReplay(n_envs, cfg.buffer_decisions, h, w, cfg, self.rng)
        self.updates = 0
        self.beta0 = self.beta = cfg.per_beta

    def epsilon(self, decisions: int, budget: int) -> float:
        c = self.cfg
        frac = min(1.0, decisions / max(1, c.eps_fraction * budget))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def act(self, stacks: np.ndarray, eps: float) -> np.ndarray:
        if self.cfg.noisy and self.q.training:
            self.q.reset_noise()  # otherwise one frozen draw acts until learning starts
        q = self.q.q(torch.from_numpy(stacks).to(self.device)).argmax(1).cpu().numpy()
        rand = self.rng.random(len(q)) < eps
        return np.where(rand, self.rng.integers(0, self.n_actions, len(q)), q)

    def update(self, progress: float) -> torch.Tensor:
        c = self.cfg
        if c.per:
            self.beta = self.beta0 + progress * (1 - self.beta0)
        s, a, ret, s2, done, gn, w, flat = self.replay.sample(c.batch_size, self.beta)
        dev = self.device
        s, s2 = torch.from_numpy(s).to(dev), torch.from_numpy(s2).to(dev)
        a, ret = torch.from_numpy(a).to(dev), torch.from_numpy(ret).to(dev)
        done, gn, w = (
            torch.from_numpy(done).to(dev),
            torch.from_numpy(gn).to(dev),
            torch.from_numpy(w).to(dev),
        )
        if c.noisy:
            self.q.reset_noise()
            self.target.reset_noise()
        b = torch.arange(len(a), device=dev)
        both = self.q(torch.cat([s, s2]))  # one online pass for the prediction and the Double-DQN argmax
        online, online_next = both[: len(a)], both[len(a) :].detach()
        with torch.no_grad():
            q_next = (online_next.exp() * self.q.support).sum(2) if c.distributional else online_next
            best = (q_next if c.double else self.target.q(s2)).argmax(1)
            if c.distributional:
                next_p = self.target(s2).exp()[b, best]  # [B, atoms]
                tz = (ret[:, None] + gn[:, None] * self.q.support[None] * (~done)[:, None]).clamp(c.v_min, c.v_max)
                dz = (c.v_max - c.v_min) / (c.n_atoms - 1)
                pos = (tz - c.v_min) / dz
                lo, hi = pos.floor().long(), pos.ceil().long()
                target = torch.zeros_like(next_p)
                target.scatter_add_(1, lo, next_p * (hi.float() + (lo == hi).float() - pos))
                target.scatter_add_(1, hi, next_p * (pos - lo.float()))
            else:
                target = ret + gn * self.target.q(s2)[b, best] * (~done)
        if c.distributional:
            logp = online[b, a]
            per_sample = -(target * logp).sum(1)
            err = per_sample.detach()
        else:
            pred = online[b, a]
            per_sample = F.smooth_l1_loss(pred, target, reduction="none") if c.huber else (pred - target) ** 2
            err = (pred - target).detach()
        loss = (per_sample * w).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip)
        self.opt.step()
        self.replay.update_priorities(flat, err.cpu().numpy())
        self.updates += 1
        if self.updates % c.target_every == 0:
            self.target.load_state_dict(self.q.state_dict())
        return loss.detach()  # a tensor: no GPU sync per update
