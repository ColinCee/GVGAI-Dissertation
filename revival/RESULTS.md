# GVGAI DQN dissertation, 2018 → 2026: results

What happened when the 2018 dissertation (DQN and its improvements on four GVGAI games) was rebuilt on today's software and hardware, re-run exactly, and then compared with today's methods. Full tables: [results/summary.md](results/summary.md) (regenerate with `uv run python report.py`). Every speed change: [IMPROVEMENTS.md](IMPROVEMENTS.md). Setup and caveats: [README.md](README.md).

## Headlines

1. **Same agent, same 1M game ticks: ~12 h in 2018 → ~25 min today** in practice (23–34 min for the three Aliens runs, on a machine shared with other runs), **~11 min** on a quiet machine (benchmark). Modern agents do the same budget in **3–5 min**.
2. **The 2018 results reproduce.** The replayed final agent lands on the report's numbers (Aliens learning curve 8 / 30 / 57 / 69% wins per 500 games vs the report's 4 / 22 / 58 / 68; Wafer Thin Mints 10.2 / 35% vs 10.2 / 33%), and so does the DQN → DDQN ablation. MCTS at 40 ms per move matches 2018 on all four games (e.g. Aliens 77.8 / 100% vs 79.2 / 100%, Wafer 9.7 / 75% vs 9.9 / 75.5%).
3. **At 2018's budget, the biggest single gain is a bigger network.** With the deeper residual network from IMPALA (2018) in place of 2015's three-layer one, Rainbow goes from 43.2 / 66% to **54.7 / 85%** on Aliens, beating the 2018 agent (50.1 / 81%) with the plain score as reward. DQN gains as much (39.5 / 57% → 52.7 / 73%).
4. **The 2018 reward's −0.5 penalty is what makes it work, and it costs elsewhere.** The penalty alone gives 81% wins; louder (±10) or fading versions, loss-only, or a longer look-ahead don't come close. With enough training the clean ±1 outcome catches up (92% at 5M ticks). But the penalty fits the training levels: on held-out levels DQN + 2018 reward falls to 31% (23% at 5M), and on Wafer Thin Mints it teaches survival over score (91–97% wins, score 2–3 of 10).
5. **Best result per game (10M-tick budget, 100 games):** Aliens 58.0 / 95% (Rainbow), Deceptizelda 2.0 / 100% (solved by every modern method; PPO solves it at 2018's budget). Wafer and Chase did not improve with more training: Wafer trades score for wins depending on the reward, and **Chase stays unsolved by every agent and budget** (0% wins, 1.7 points after 20M ticks).
6. **MCTS still beats every learning agent on score** (Aliens 77.8 vs 58–62), with no training.

## What was compared

All rows: 50 evaluation games, greedy, levels 0–2 (Deceptizelda: 0 and 2, see caveats). 2026 rows are the mean of 3 seeds; the range is win % across seeds. "2018's ticks" = the game ticks 2018 used for that game.

| Agent | Aliens | Chase | Deceptizelda | Wafer Thin Mints |
|---|---|---|---|---|
| Human (2018) | 42.6 (100%) | 1.2 (0%) | 10.6 (80%) | 11 (0%) |
| MCTS, 40 ms per move (2018) | 79.2 (100%) | 3.5 (3.7%) | 1.91 (84.4%) | 9.9 (75.5%) |
| 2018 final agent (2018 report) | 56.3 (82.6%) | 0.7 (0%) | 1.99 (99%) | 10.2 (33%) |
| MCTS, 40 ms per move (2026 PC) | 77.8 (100%) | 3.5 (10%) | 1.9 (90%) | 9.7 (75%) |
| 2018 final agent, replayed | 50.1 (81%, 66–92) | 0.2 (0%) | 1.6 (79%, 38–100) | 10.2 (35%, 32–38) |
| Modern DQN, 2018's ticks | 39.5 (57%, 42–84) | 0.0 (0%) | 1.3 (67%, 0–100) | 7.7 (52%, 34–70) |
| Modern DQN + 2018's reward\*, 2018's ticks | 39.4 (85%, 80–88) | – | – | 7.5 (59%, 32–86) |
| Rainbow + 2018's reward\*, 2018's ticks | 39.6 (88%, 82–94) | 0.0 (0%) | 2.0 (100%) | 2.7 (91%, 76–100) |
| Rainbow, 2018's ticks | 43.2 (66%, 62–72) | 0.0 (0%) | 1.0 (42%, 0–100) | 7.4 (57%, 42–86) |
| Rainbow, IMPALA network, 2018's ticks | 54.7 (85%, 82–90) | | | |
| PPO, 2018's ticks | 37.6 (82%, 78–86) | 0.0 (0%) | 2.0 (100%) | 6.3 (73%, 72–74) |
| Modern DQN, up to 5M ticks | 58.6 (89%, 84–94) | 0.6 (0%) | 2.0 (100%) | 9.5 (41%, 38–44) |
| Rainbow, up to 5M ticks | 57.0 (93%, 90–98) | 0.0 (0%) | 2.0 (100%) | 9.5 (35%, 20–60) |
| Best 5M setup, up to 10M ticks (100 games) | 58.0 (95%, 93–98) | 0.8 (0%) | 2.0 (100%) | 10.2 (31%, 24–37) |

\* 2018's reward = ±1 for the game's outcome plus −0.5 on every move of a lost game: outcome shaping that uses knowledge of the game, not part of the standard methods.

## The reward question (Aliens, modern DQN, 2018's ticks, 3 seeds)

| Reward | Score | Wins | Held-out levels |
|---|---|---|---|
| Score changes only (standard) | 39.5 | 57% | 61% |
| + ±1 outcome at game end | 40.5 | 45% | 31% |
| + ±5 / ±10 outcome | 36.5 / 40.7 | 54% / 61% | 54% / 36% |
| + −1 on a loss only | 40.6 | 44% | 57% |
| + ±1 outcome, longer look-ahead (γ 0.997) | 38.1 | 32% | 29% |
| + −0.5 on every move of a lost game, no outcome | 38.5 | 81% | 49% |
| + the same penalty, fading with distance from the loss | 41.6 | 75% | 41% |
| **2018's reward (±1 outcome + −0.5 penalty)** | 39.4 | **85%** | 31% |
| 2018's reward, penalty −1 | 40.9 | 90% | 69% |
| 2018's reward, 2018's smaller replay (100k) | 42.3 | 88% | 78% |
| 2018 agent itself | 50.1 | 81% | – |
| ±1 outcome only, 5M ticks | 57.9 | 92% | 72% |
| 2018's reward, 5M ticks | 48.2 | 91% | 23% |

- **The per-move penalty is the part that works.** It puts the loss signal on every move of the lost game, where a single end-of-game number has to be passed back ~150 moves and competes with dozens of +1s for kills. A louder end signal doesn't fix that; more training does (92% at 5M).
- **It overfits.** Held-out wins fall from 61% (score only) to 31%, and to 23% after 5M ticks. Oddly, a smaller replay buffer or a bigger penalty keeps held-out wins up (78%, 69%); one seed set each, so treat as a lead, not a result.
- **It changes the strategy, not only the win rate.** On Aliens the won-game score stays below the 2018 agent's (42.7 vs 54.3: the 2018 agent also shoots the protective bases). On Wafer Thin Mints, where surviving to tick 1500 is a win, Rainbow and PPO with the penalty learn to survive and eat little: 91–97% wins at 2–3 points, against ~35% wins at ~10 points without it.

## Network, algorithm and training budget (Aliens)

| Change | 2018's ticks: score (wins) | Held-out | Up to 5M ticks |
|---|---|---|---|
| Modern DQN | 39.5 (57%) | 61% | 58.6 (89%) |
| DQN, IMPALA network | 52.7 (73%) | 71% | 62.0 (98%), 1 seed\* |
| Rainbow | 43.2 (66%) | 89% | 57.0 (93%) |
| Rainbow, IMPALA network | **54.7 (85%)** | 78% | 57.6 (90%), 1 seed\* |
| Rainbow, IMPALA + 2018's reward | 43.4 (95%) | 69% | – |
| PPO (policy gradient, not Q-learning) | 37.6 (82%) | 72% | 45.9 (83%) |
| DQN, 2× / 4× more updates per tick | 47.2 (35%) / 53.0 (70%) | 57% / 46% | – |
| Rainbow, 2× / 4× more updates per tick | 46.8 (63%) / 51.0 (65%) | 73% / 65% | – |
| DQN / Rainbow trained on all 5 levels | – | – | 55.8 (79%) / 56.0 (81%); "held-out" 83% / 77%, but those levels were seen |
| Rainbow to 10M ticks (stopped at ~7.7M) | – | – | 58.0 (95%), held-out 80% |

\* Seeds 1–2 were still training at the time of writing.

- **The network matters most at small budgets**: +11 points on both DQN and Rainbow. More updates per tick also raise the score (+8–14) but make wins unstable.
- **PPO** is the steadiest without shaping: 82% wins at 2018's budget, 72% on held-out levels. It scores lower (it plays safe) and solves Deceptizelda at 2018's budget.
- **Beyond ~5M ticks Aliens plateaus** around 57–62 points, still well short of MCTS's 78.

## Reproducing 2018 (Aliens, win % per 500 training games)

| Variant | 2018 report | 2026 replay |
|---|---|---|
| DQN | 6 / 12 / 29 / 35 | 6 / 12 / 26 / 31 |
| DDQN | 3 / 17 / 47 / 57 | 8 / 19 / 44 / 59 |
| Final (DDQN + PER + new reward) | 4 / 22 / 58 / 68 | 8 / 30 / 57 / 69 |

Two bugs in the 2018 code (rewards credited one move late; prioritised replay without importance weights), fixed and nothing else: 48.2 / 80% from one seed, no better than the replay, so they probably didn't matter much.

## Today's methods, one piece at a time (Aliens, 2018's ticks, seed 0)

| Step | Score (wins) |
|---|---|
| Modern DQN | 39.5 (57%) |
| + prioritised replay | 40.1 (62%) |
| + dueling | 43.0 (50%) |
| + 3-step returns | 46.2 (82%) |
| + noisy nets + distributional (= Rainbow) | 43.2 (66%) |

Suggestive only (one seed, ±10 points): 3-step returns look like the piece that helps at this short budget; the rest is within noise (dueling even lands below the baseline).

## Speed

| | 2018 | 2026 |
|---|---|---|
| One game, engine alone, ticks/s | ≤40 (a 25 ms pause every tick was built in; 34 measured) | ~2,600 |
| Training, ticks/s | ~23 (1M ticks in ~12 h) | 2018 agent ~600–1,500 per run; modern agents ~9,000 per run, ~16,000 with 3 at once |
| One 1M-tick run | ~12 h | 3–11 min |
| This whole project's training (241 runs, 554M ticks) | ~280 days of non-stop training | ~15 h of wall clock (39 h of run time, 3 runs at once) |

## Caveats

- One seed per agent in 2018 versus 3 here; 2018's own two runs of the same agent differ by 10–17 win-% points, and a 50-game evaluation is ±10 points.
- Deceptizelda's level 1 is a bigger map than 2018's network could have played; all agents use levels 0 and 2.
- Chase's only 2018 run used the old reward (no penalty); the replay does too.
- Wafer Thin Mints: the engine's current rules differ from 2018's; the 2018 rules were rebuilt from the engine's history.
- Modern agents see 2×2-averaged greyscale frames; the 2018 replay keeps 2018's PIL Lanczos frames.
- "Lost" for the −0.5 penalty is the game's own verdict, as in 2018's code: on Wafer Thin Mints surviving to tick 1500 is a win, dying a loss.
- MCTS at 40 ms on today's CPU gets far more search per move than in 2018 (140–600 iterations per move), yet scores the same: the games are easy for it either way.
- The 2018 agent's timing averages one CPU run and two GPU runs.

## Still running

Seeds 1–2 of the IMPALA network to 5M ticks (Aliens), and MCTS at fixed search budgets (50 / 200 / 800 iterations per move; 2026's 40 ms buys 140–600).
