"""Modern value-based agent, from DQN-with-today's-defaults up to Rainbow, one switch per improvement.

The Rainbow parts (noisy nets, C51 projection, PER with importance weights) follow CleanRL's
rainbow_atari.py (MIT, https://github.com/vwxyzjn/cleanrl). Differences that matter here:
- many games at once (gvgai/vec.py); replay keeps one frame per decision per game and rebuilds
  stacks of 4 at sampling time, so 1M decisions take ~8 GB rather than ~33 GB;
- each decision repeats the action for `action_repeat` ticks (Atari's default skip of 4;
  2018 also acted every 4 ticks); budgets are counted in game ticks.
"""

import math
import os
import tempfile
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
    buffer_decisions: int = 500_000  # 2M ticks of history; 3 runs fit on the 4090 at once (see IMPROVEMENTS 23)
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
    # Outcome reward: a game's last move gets +1 for a win or -1 otherwise (timeouts count as losses, as
    # in 2018), replacing its score change. loss_penalty: also subtract this from every move of a lost
    # game, clipped at -1 (2018 used 0.5). Off by default: modern agents normally see only the score.
    outcome_reward: bool = False
    win_reward: float = 1.0  # outcome values when outcome_reward is on
    loss_reward: float = -1.0
    loss_penalty: float = 0.0
    score_scale: float = 1.0  # weight of score changes; 0 with outcome_reward: the outcome is the only reward
    penalty_decay: float = 1.0  # <1: a move k moves before the loss gets loss_penalty * decay**k
    # Saturating win reward: >0 makes a win worth 1 - 0.5 * c / (c + game score), with c this value
    # (random play's score on the game), so every win is worth 0.5 to 1 and beats any loss, more
    # points always help, and each extra point helps less. No score cap needed.
    win_score_ref: float = 0.0
    grad_clip: float = 10.0
    compile: bool = True  # torch.compile the learner's networks on the GPU
    action_repeat: int = 4
    amp: bool = True  # bf16 network passes on the GPU
    scale_input: bool = True  # pixels / 255 (2018 fed raw 0-255 values)
    encoder: str = "nature"  # or "impala"


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


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.c1, self.c2 = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.c2(F.relu(self.c1(F.relu(x))))


def impala_encoder(channels=(16, 32, 32)) -> nn.Sequential:
    """IMPALA's 15-layer residual encoder (Espeholt et al. 2018), standard on Procgen."""
    layers: list[nn.Module] = []
    c_in = STACK
    for c in channels:
        layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.MaxPool2d(3, 2, padding=1), ResBlock(c), ResBlock(c)]
        c_in = c
    return nn.Sequential(*layers, nn.ReLU(), nn.Flatten())


class Net(nn.Module):
    def __init__(self, h: int, w: int, n_actions: int, cfg: ModernConfig):
        super().__init__()
        self.cfg, self.n_actions = cfg, n_actions
        self.atoms = cfg.n_atoms if cfg.distributional else 1
        if cfg.encoder == "impala":
            self.conv = impala_encoder()
        else:  # the Nature DQN encoder 2018 used
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
        # bf16 for the layers on the GPU (fp32 accumulate); combining and softmax stay in fp32
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg.amp and x.is_cuda):
            x = x.float() / 255.0 if self.cfg.scale_input else x.float()
            h = self.conv(x.contiguous(memory_format=torch.channels_last))
            a = self.adv(h)
            v = self.val(h) if self.val is not None else None
        a = a.float().view(-1, self.n_actions, self.atoms)
        if v is not None:
            a = v.float().view(-1, 1, self.atoms) + a - a.mean(1, keepdim=True)
        return F.log_softmax(a, 2) if self.cfg.distributional else a.squeeze(2)

    def q(self, x: torch.Tensor) -> torch.Tensor:
        out = self(x)
        return (out.exp() * self.support).sum(2) if self.cfg.distributional else out

    def reset_noise(self) -> None:
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


def upload(arr: np.ndarray, device: str) -> torch.Tensor:
    """Copy to the GPU through pinned memory, so the CPU doesn't wait for queued GPU work.

    From ordinary memory a copy blocks until everything queued before it (the learner's updates)
    has run. torch's pinned-memory cache makes the staging buffer cheap to get and safe to reuse.
    """
    if not str(device).startswith("cuda"):
        return torch.from_numpy(arr)
    staging = torch.empty(arr.shape, dtype=torch.from_numpy(arr[:0]).dtype, pin_memory=True)
    staging.numpy()[...] = arr
    return staging.to(device, non_blocking=True)


class GpuReplay:
    """Replay held on the GPU: per game, a ring of single frames, one slot per decision.

    A slot holds the frame a decision was made on, its action, reward and flags. When a game ends,
    its final frame goes into the next slot as a frame-only entry, so a transition's next frame is
    always the next slot's. Stacks of 4 are rebuilt at sampling time, padded at an episode start
    with its first frame (as `Stacks` does when acting). Plain and prioritised sampling share one
    path: per-slot weights (1, or priority**alpha) masked to valid slots, cumulative sum, search.
    """

    def __init__(self, n_envs: int, capacity: int, h: int, w: int, cfg: ModernConfig, device: str):
        n, cap = n_envs, capacity // n_envs
        self.n, self.cap, self.cfg, self.dev = n, cap, cfg, device
        z = dict(device=device)
        self.frames = torch.zeros((n, cap, h, w), dtype=torch.uint8, **z)
        self.action = torch.zeros((n, cap), dtype=torch.long, **z)
        self.reward = torch.zeros((n, cap), **z)
        self.term = torch.zeros((n, cap), dtype=torch.bool, **z)  # game over: no bootstrap
        self.end = torch.zeros((n, cap), dtype=torch.bool, **z)  # episode ended (game over or timeout)
        self.start = torch.zeros((n, cap), dtype=torch.long, **z)  # absolute index of the episode's first slot
        self.weight = torch.zeros(n * cap, **z)  # 0 for frame-only slots
        self.max_p = torch.ones((), **z)
        self.t = torch.zeros(n, dtype=torch.long, **z)  # slots written per game (absolute)
        self.t_host = np.zeros(n, np.int64)
        self.ep_start = np.zeros(n, np.int64)
        self.ep_score = np.zeros(n, np.float64)  # raw score so far this episode, for win_score_ref
        self.slot_ids = torch.arange(cap, **z)
        self.offs = torch.arange(-STACK + 1, 1, **z)

    def add(self, obs: np.ndarray, action, reward, term, trunc, final_obs: np.ndarray, won=None) -> None:
        """obs[e]: the frame action[e] was chosen on; final_obs[e]: the frame after an episode ended;
        won[e]: whether an ended episode was won (needed for outcome_reward and loss_penalty)."""
        es, ts, frames, acts, rews, terms, ends, starts, fo = [], [], [], [], [], [], [], [], []
        lost: list[tuple[int, int, int]] = []  # (game, first, last) absolute slots of lost episodes
        for e in range(self.n):
            ended = bool(term[e] or trunc[e])
            es.append(e), ts.append(self.t_host[e]), frames.append(obs[e]), acts.append(int(action[e]))
            self.ep_score[e] += float(reward[e])
            r = self.cfg.score_scale * (float(np.sign(reward[e])) if self.cfg.clip_reward else float(reward[e]))
            if self.cfg.outcome_reward and ended:
                # 2018 replaced the last move's score change with the outcome (+1 / -1)
                r = self.cfg.win_reward if won[e] else self.cfg.loss_reward
                if won[e] and self.cfg.win_score_ref > 0:
                    c = self.cfg.win_score_ref
                    r = 1.0 - 0.5 * c / (c + max(self.ep_score[e], 0.0))
            if self.cfg.loss_penalty and ended and not won[e]:  # a loss is the game's own: death or timeout
                lost.append((e, int(self.ep_start[e]), int(self.t_host[e])))
            rews.append(r), terms.append(bool(term[e])), ends.append(ended), starts.append(self.ep_start[e])
            fo.append(False)
            self.t_host[e] += 1
            if ended:
                es.append(e), ts.append(self.t_host[e]), frames.append(final_obs[e]), acts.append(0), rews.append(0.0)
                terms.append(False), ends.append(False), starts.append(self.ep_start[e]), fo.append(True)
                self.t_host[e] += 1
                self.ep_start[e] = self.t_host[e]
                self.ep_score[e] = 0.0
        # one upload for all the small fields (separate torch.tensor calls cost ~0.2 ms each)
        dev = self.dev
        meta = upload(np.array([es, ts, acts, starts, terms, ends, fo], np.int64), dev)
        e_t, k_t = meta[0], meta[1] % self.cap
        fo_t = meta[6].bool()
        self.frames[e_t, k_t] = upload(np.stack(frames), dev)
        self.action[e_t, k_t] = meta[2]
        self.reward[e_t, k_t] = upload(np.array(rews, np.float32), dev)
        self.term[e_t, k_t] = meta[4].bool()
        self.end[e_t, k_t] = meta[5].bool()
        self.start[e_t, k_t] = meta[3]
        p_new = self.max_p**self.cfg.per_alpha if self.cfg.per else torch.ones((), device=dev)
        self.weight[e_t * self.cap + k_t] = torch.where(fo_t, torch.zeros((), device=dev), p_new)
        self.t.copy_(upload(self.t_host, dev), non_blocking=True)
        for e, first, last in lost:  # 2018's penalty on every move of a lost game (this one included)
            first = max(first, last - self.cap + 1)
            ks = torch.arange(first, last + 1, device=dev)
            amount = self.cfg.loss_penalty * self.cfg.penalty_decay ** (last - ks).float()
            k = ks % self.cap
            low = min(-1.0, self.cfg.loss_reward)  # 2018 clipped at -1; a louder loss keeps its value
            self.reward[e, k] = torch.maximum(self.reward[e, k] - amount, torch.full_like(amount, low))

    def _stack(self, e: torch.Tensor, abs_t: torch.Tensor) -> torch.Tensor:
        ts = abs_t[:, None] + self.offs[None, :]
        ts = torch.maximum(ts, self.start[e, abs_t % self.cap][:, None])
        return self.frames[e[:, None], ts % self.cap]

    def sample(self, k: int, beta: torch.Tensor):
        cfg, cap = self.cfg, self.cap
        if not cfg.per:
            return self._sample_uniform(k)
        t = self.t[:, None]
        abs_all = t - 1 - ((t - 1 - self.slot_ids[None, :]) % cap)
        # history of STACK frames still in the ring, and n_step decisions (plus the next frame) after it
        valid = (abs_all >= t - cap + STACK) & (abs_all <= t - 1 - cfg.n_step) & (abs_all >= 0)
        w_all = (self.weight * valid.flatten()).double()
        cdf = w_all.cumsum(0)  # float64: float32 partial sums over 1M slots break the search
        total = cdf[-1]
        u = torch.arange(k, device=self.dev, dtype=torch.float64) + torch.rand(k, device=self.dev, dtype=torch.float64)
        flat = torch.searchsorted(cdf, u * total / k).clamp(max=self.n * cap - 1)
        e, abs_t = flat // cap, abs_all.flatten()[flat]
        p = w_all[flat] / total
        wts = (valid.sum() * p) ** (-beta)
        return self._finish(k, e, abs_t, (wts / wts.max()).float(), flat)

    def _finish(self, k: int, e: torch.Tensor, abs_t: torch.Tensor, wts: torch.Tensor, flat: torch.Tensor):
        """Stacks, n-step returns and next stacks for the chosen slots."""
        cfg, cap = self.cfg, self.cap
        s = self._stack(e, abs_t)
        ret = torch.zeros(k, device=self.dev)
        done = torch.zeros(k, dtype=torch.bool, device=self.dev)
        alive = torch.ones(k, dtype=torch.bool, device=self.dev)
        steps = torch.zeros(k, device=self.dev)
        last = abs_t.clone()
        for j in range(cfg.n_step):
            kj = (abs_t + j) % cap
            ret += torch.where(alive, (cfg.gamma**j) * self.reward[e, kj], 0.0)
            steps += alive
            last = torch.where(alive, abs_t + j, last)
            done |= self.term[e, kj] & alive
            alive &= ~self.end[e, kj]
        s2 = self._stack(e, last + 1)  # after an episode end, last + 1 is its frame-only slot
        a = self.action[e, abs_t % cap]
        return s, a, ret, s2, done, cfg.gamma**steps, wts, flat

    def _sample_uniform(self, k: int):
        """Plain replay without scanning every slot: draw 2k candidates uniformly inside each game's
        valid window and keep the first k that aren't frame-only slots (~1 in an episode's length, so
        a miss among 2k draws is ~1e-4; a miss gets weight 0)."""
        cfg, cap, dev = self.cfg, self.cap, self.dev
        lo = (self.t - cap + STACK).clamp(min=0)
        hi = self.t - 1 - cfg.n_step
        e = torch.randint(0, self.n, (2 * k,), device=dev)
        span = (hi - lo + 1).clamp(min=1)[e]
        abs_c = lo[e] + (torch.rand(2 * k, device=dev) * span).long()
        ok = self.weight[e * cap + abs_c % cap] > 0
        pick = torch.argsort((~ok).to(torch.int8), stable=True)[:k]
        e, abs_t, ok = e[pick], abs_c[pick], ok[pick]
        return self._finish(k, e, abs_t, ok.float(), e * cap + abs_t % cap)

    def update_priorities(self, flat: torch.Tensor, err: torch.Tensor) -> None:
        if not self.cfg.per:
            return
        p = err.abs() + self.cfg.per_eps
        self.max_p.copy_(torch.maximum(self.max_p, p.max()))  # in place: also runs inside the CUDA graph
        self.weight[flat] = p**self.cfg.per_alpha


class ModernAgent:
    def __init__(self, n_envs: int, h: int, w: int, n_actions: int, cfg: ModernConfig, device: str, seed: int):
        self.cfg, self.device, self.n_actions = cfg, device, n_actions
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        # channels-last weights and inputs, and cuDNN's algorithm search: ~10% off each update
        torch.backends.cudnn.benchmark = True
        self.q = Net(h, w, n_actions, cfg).to(device, memory_format=torch.channels_last)
        self.target = Net(h, w, n_actions, cfg).to(device, memory_format=torch.channels_last)
        self.target.load_state_dict(self.q.state_dict())
        cuda = str(device).startswith("cuda")
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr, eps=cfg.adam_eps, fused=cuda, capturable=cuda)
        if cuda and cfg.compile:
            # Fuses the small steps between layers into fewer kernels, inside the CUDA graph below (which
            # already removes launch costs): Rainbow update 1.4x, IMPALA 1.2x, Nature 1.1x. Not autotuned:
            # that adds 12% on IMPALA but its first compile can pass the engine's 500 s limit on one move.
            # Each process its own compile cache: runs starting together raced on the shared one and loaded
            # half-written kernels.
            os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", tempfile.mkdtemp(prefix="inductor-"))
            # and Triton's kernel cache too: a Rainbow + IMPALA run loaded a half-written one (25 Oct 2026)
            os.environ.setdefault("TRITON_CACHE_DIR", tempfile.mkdtemp(prefix="triton-"))
            for net in (self.q, self.target):
                net.compile()
        self.replay = GpuReplay(n_envs, cfg.buffer_decisions, h, w, cfg, device)
        self.updates = 0
        self.beta0 = cfg.per_beta
        self.beta = torch.full((), cfg.per_beta, device=device)
        # On the GPU the whole update (sampling included) is recorded once as a CUDA graph and replayed:
        # launching its ~100 small kernels one by one from Python cost ~4x the GPU time itself.
        self.use_graph = cuda
        self.graph: torch.cuda.CUDAGraph | None = None
        self.loss = torch.zeros((), device=device)
        # Acting runs on its own stream with one of two copies of the network, refreshed after each
        # step's updates, so choosing a move never waits behind the learner's queued updates (the
        # copy used is the one refreshed two steps ago: its weights are at most ~2 steps old).
        self.act_stream = torch.cuda.Stream(priority=-1) if cuda else None
        self.actors = (
            [Net(h, w, n_actions, cfg).to(device, memory_format=torch.channels_last) for _ in range(2)] if cuda else []
        )
        for net in self.actors:
            net.load_state_dict(self.q.state_dict())
            net.requires_grad_(False)
        self.actor_ready: list[torch.cuda.Event | None] = [None, None]
        self.steps = 0

    def refresh_actor(self) -> None:
        """After this step's updates are queued: copy the online net into this step's actor slot."""
        if not self.actors:
            return
        i = self.steps % 2
        src = list(self.q.parameters()) + list(self.q.buffers())
        dst = list(self.actors[i].parameters()) + list(self.actors[i].buffers())
        with torch.no_grad():
            torch._foreach_copy_(dst, src)
        ev = torch.cuda.Event()
        ev.record()
        self.actor_ready[i] = ev
        self.steps += 1

    def epsilon(self, decisions: int, budget: int) -> float:
        c = self.cfg
        frac = min(1.0, decisions / max(1, c.eps_fraction * budget))
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    @torch.no_grad()
    def act(self, stacks: np.ndarray | torch.Tensor, eps: float, actor: bool = False) -> np.ndarray:
        """`actor=True` (training loop on the GPU): use the actor copy on the acting stream."""
        if actor and self.actors:
            i = self.steps % 2
            with torch.cuda.stream(self.act_stream):
                if self.actor_ready[i] is not None:
                    self.act_stream.wait_event(self.actor_ready[i])
                net = self.actors[i]
                if self.cfg.noisy:
                    net.reset_noise()
                x = stacks if isinstance(stacks, torch.Tensor) else upload(stacks, self.device)
                q = net.q(x).argmax(1).cpu().numpy()
        else:
            if self.cfg.noisy and self.q.training:
                self.q.reset_noise()  # otherwise one frozen draw acts until learning starts
            x = stacks if isinstance(stacks, torch.Tensor) else upload(stacks, self.device)
            q = self.q.q(x).argmax(1).cpu().numpy()
        rand = self.rng.random(len(q)) < eps
        return np.where(rand, self.rng.integers(0, self.n_actions, len(q)), q)

    def update(self, progress: float) -> torch.Tensor:
        c = self.cfg
        if c.per:
            self.beta.fill_(self.beta0 + progress * (1 - self.beta0))
        if not self.use_graph:
            self._update_body()
        elif self.graph is not None:
            self.graph.replay()
        else:
            # warm up on a side stream (cuDNN/cuBLAS set-up must not be captured), then record
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._update_body()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            self.opt.zero_grad(set_to_none=True)
            with torch.cuda.graph(self.graph):
                self._update_body(zero_grad=False)
            self.graph.replay()
        self.updates += 1
        if self.updates % c.target_every == 0:
            self.target.load_state_dict(self.q.state_dict())  # copies in place: the graph sees it
        return self.loss

    def _update_body(self, zero_grad: bool = True) -> None:
        """One gradient update; only tensor ops on fixed shapes so it can be captured."""
        c = self.cfg
        s, a, ret, s2, done, gn, w, flat = self.replay.sample(c.batch_size, self.beta)
        dev = self.device
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
        if zero_grad:
            self.opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), c.grad_clip, foreach=True)
        self.opt.step()
        self.replay.update_priorities(flat, err)
        self.loss.copy_(loss.detach())
