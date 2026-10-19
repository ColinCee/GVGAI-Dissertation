"""PPO on one game: the policy-gradient branch of today's methods, for comparison with the DQN family.

    uv run python train_ppo.py --game aliens --ticks 1014134 --seed 0 --name aliens-ppo-b18-s0

Follows CleanRL's ppo_atari.py defaults (8 games x 128 moves per rollout, 4 epochs of 4 minibatches,
clip 0.1, entropy 0.01, GAE 0.95, learning rate annealed to 0) on the same env layer, preprocessing,
action repeat and evaluation as train_modern.py. A timeout is a truncation: its value is bootstrapped.
"""

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agents.modern import STACK, upload
from gvgai import heartbeat
from gvgai.games import HELD_OUT, game_levels, git_commit
from gvgai.vec import vector
from train_modern import Stacks, evaluate, finished

HERE = Path(__file__).resolve().parent


@dataclass
class PPOConfig:
    steps: int = 128  # moves per game per rollout
    epochs: int = 4
    minibatches: int = 4
    lr: float = 2.5e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    action_repeat: int = 4
    outcome_reward: bool = False  # the dissertation's reward, as in agents/modern.py
    score_scale: float = 1.0  # weight of score changes; 0: the only reward is the game's outcome at its end
    loss_penalty: float = 0.0


class ActorCritic(nn.Module):
    def __init__(self, h: int, w: int, n_actions: int):
        super().__init__()

        def layer(m: nn.Module, std: float = 2**0.5) -> nn.Module:
            nn.init.orthogonal_(m.weight, std)
            nn.init.zeros_(m.bias)
            return m

        self.body = nn.Sequential(
            layer(nn.Conv2d(STACK, 32, 8, 4)), nn.ReLU(),
            layer(nn.Conv2d(32, 64, 4, 2)), nn.ReLU(),
            layer(nn.Conv2d(64, 64, 3, 1)), nn.ReLU(),
            nn.Flatten(),
        )  # fmt: skip
        with torch.no_grad():
            n = self.body(torch.zeros(1, STACK, h, w)).shape[1]
        self.body.append(layer(nn.Linear(n, 512)))
        self.body.append(nn.ReLU())
        self.pi = layer(nn.Linear(512, n_actions), 0.01)
        self.v = layer(nn.Linear(512, 1), 1.0)

    def forward(self, x: torch.Tensor):
        h = self.body(x.float() / 255.0)
        return self.pi(h), self.v(h).squeeze(-1)


class GreedyAgent:
    """What train_modern.evaluate expects: .q with eval()/train(), and act(stacks, eps)."""

    def __init__(self, net: ActorCritic, device: str):
        self.q, self.device = net, device

    @torch.no_grad()
    def act(self, stacks, eps: float, actor: bool = False) -> np.ndarray:
        x = stacks if isinstance(stacks, torch.Tensor) else upload(stacks, self.device)
        return self.q(x)[0].argmax(1).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="aliens")
    p.add_argument("--ticks", type=int, default=1_014_134)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", required=True)
    p.add_argument("--reward2018", action="store_true", help="the dissertation's outcome reward and -0.5 penalty")
    p.add_argument("--outcome-only", action="store_true", help="reward only the outcome: +1 win, -1 loss, at game end")
    p.add_argument("--outcome-score", type=float, default=None, help="±1 outcome plus this weight on score changes")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--eval-games", type=int, default=50)
    a = p.parse_args()
    commit = git_commit()
    scale = 0.0 if a.outcome_only else a.outcome_score if a.outcome_score is not None else 1.0
    cfg = PPOConfig(outcome_reward=a.reward2018 or a.outcome_only or a.outcome_score is not None, score_scale=scale,
                    loss_penalty=0.5 if a.reward2018 else 0.0)  # fmt: skip
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    dev = a.device
    run = HERE / "runs" / a.name
    run.mkdir(parents=True, exist_ok=True)
    levels = game_levels(a.game)
    env = vector(a.game, a.envs, a.seed, levels, cfg.action_repeat)
    obs, _ = env.reset(seed=a.seed)
    n_actions = int(env.single_action_space.n)
    h, w = obs.shape[1:]
    net = ActorCritic(h, w, n_actions).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    stacks = Stacks(obs, dev)
    n, T = a.envs, cfg.steps
    buf_s = torch.zeros((T, n, STACK, h, w), dtype=torch.uint8, device=dev)
    buf_a = torch.zeros((T, n), dtype=torch.long, device=dev)
    buf_logp, buf_v, buf_r, buf_done = (torch.zeros((T, n), device=dev) for _ in range(4))
    ep_moves: list[list[int]] = [[] for _ in range(n)]  # rollout slots of each game's current episode
    ticks = episodes = updates = 0
    total_rollouts = max(1, a.ticks // (cfg.action_repeat * n * T))
    t0 = time.perf_counter()
    with open(run / "episodes.csv", "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["episode", "score", "win", "level", "cum_ticks", "updates", "epsilon", "wall_s"])
        for it in range(total_rollouts):
            heartbeat.check()
            for g in opt.param_groups:
                g["lr"] = cfg.lr * (1 - it / total_rollouts)
            ep_moves = [[] for _ in range(n)]
            for t in range(T):
                s = stacks.array()
                with torch.no_grad():
                    logits, v = net(s)
                    dist = torch.distributions.Categorical(logits=logits)
                    act = dist.sample()
                buf_s[t], buf_a[t], buf_logp[t], buf_v[t] = s, act, dist.log_prob(act), v
                obs, r, term, trunc, info = env.step(act.cpu().numpy())
                ended = term | trunc
                r = cfg.score_scale * np.sign(r).astype(np.float32)
                won = info["final_info"]["win"] if ended.any() else None
                for e in range(n):
                    ep_moves[e].append(t)
                    if not ended[e]:
                        continue
                    if cfg.outcome_reward:
                        r[e] = 1.0 if won[e] else -1.0
                    if cfg.loss_penalty and not won[e]:
                        # 2018's penalty on the lost game's moves; moves from earlier rollouts are already
                        # used, so only this rollout's part of the game gets it (an approximation)
                        k = torch.tensor(ep_moves[e][:-1], device=dev, dtype=torch.long)
                        if len(k):
                            buf_r[k, e] = (buf_r[k, e] - cfg.loss_penalty).clamp(min=-1.0)
                        r[e] = max(r[e] - cfg.loss_penalty, -1.0)
                    ep_moves[e] = []
                if trunc.any():  # timeout: bootstrap from the value of the frame the game stopped on
                    final = np.array(obs)
                    for e in np.flatnonzero(trunc):
                        final[e] = info["final_obs"][e]
                    fs = torch.cat([stacks.array()[:, 1:], upload(final, dev)[:, None]], 1)
                    with torch.no_grad():
                        r = r + cfg.gamma * trunc * net(fs)[1].cpu().numpy()
                buf_r[t] = torch.from_numpy(r.astype(np.float32)).to(dev)
                buf_done[t] = torch.from_numpy(ended.astype(np.float32)).to(dev)
                stacks.push(obs, ended)
                ticks += int(np.sum(info["ticks"]))
                for row in finished(info, ended):
                    episodes += 1
                    out.writerow([episodes, row["score"], int(row["win"]), row["level"], ticks, updates, "",
                                  f"{time.perf_counter() - t0:.1f}"])  # fmt: skip
            with torch.no_grad():  # GAE
                next_v = net(stacks.array())[1]
                adv = torch.zeros_like(buf_r)
                last = torch.zeros(n, device=dev)
                for t in reversed(range(T)):
                    nv = next_v if t == T - 1 else buf_v[t + 1]
                    nonterm = 1 - buf_done[t]
                    delta = buf_r[t] + cfg.gamma * nv * nonterm - buf_v[t]
                    last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
                    adv[t] = last
                ret = adv + buf_v
            bs, ba, blp = buf_s.reshape(T * n, STACK, h, w), buf_a.reshape(-1), buf_logp.reshape(-1)
            badv, bret = adv.reshape(-1), ret.reshape(-1)
            mb = T * n // cfg.minibatches
            for _ in range(cfg.epochs):
                perm = torch.from_numpy(rng.permutation(T * n)).to(dev)
                for k in range(cfg.minibatches):
                    i = perm[k * mb : (k + 1) * mb]
                    logits, v = net(bs[i])
                    dist = torch.distributions.Categorical(logits=logits)
                    ratio = (dist.log_prob(ba[i]) - blp[i]).exp()
                    ad = (badv[i] - badv[i].mean()) / (badv[i].std() + 1e-8)
                    pg = torch.max(-ad * ratio, -ad * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                    loss = pg + cfg.vf_coef * F.mse_loss(v, bret[i]) - cfg.ent_coef * dist.entropy().mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                    opt.step()
                    updates += 1
            if it % 20 == 0:
                f.flush()
                el = time.perf_counter() - t0
                print(f"ticks {ticks} ({ticks / el:.0f}/s) episodes {episodes} updates {updates}", flush=True)
    env.close()
    train_wall = time.perf_counter() - t0
    agent = GreedyAgent(net, dev)
    ev = evaluate(agent, a.game, a.envs, a.eval_games, levels, a.seed + 7, cfg.action_repeat)
    ev_held = evaluate(agent, a.game, a.envs, a.eval_games, HELD_OUT, a.seed + 8, cfg.action_repeat)
    agent = "ppo-reward2018" if a.reward2018 else "ppo-outcome-only" if a.outcome_only else "ppo"
    agent = f"ppo-outcome-score{a.outcome_score}" if a.outcome_score is not None else agent
    result = {
        "name": a.name, "agent": agent, "game": a.game, "seed": a.seed,
        "train_ticks": ticks, "train_wall_s": train_wall, "updates": updates, "envs": a.envs, "config": asdict(cfg),
        "eval": ev, "eval_heldout": ev_held, "budget_ticks": a.ticks, "commit": commit,
    }  # fmt: skip
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{a.name}.json").write_text(json.dumps(result, indent=1))
    print(f"done: {ticks} ticks in {train_wall / 60:.1f} min; eval {ev['mean_score']:.1f} ({ev['win_rate']:.0%})")


if __name__ == "__main__":
    main()
    os._exit(0)  # see train_modern.py
