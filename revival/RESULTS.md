# GVGAI DQN dissertation, 2018 → 2026: results

What happened when the 2018 dissertation (DQN and its improvements on four GVGAI games) was rebuilt on today's software and hardware, re-run exactly, and then compared with today's methods. Full tables: [results/summary.md](results/summary.md) (regenerate with `uv run python report.py`). Every speed change: [IMPROVEMENTS.md](IMPROVEMENTS.md). Setup and caveats: [README.md](README.md).

## Headlines

1. **Same agent, same 1M game ticks: ~12 h in 2018 → ~25 min today** in practice (23–34 min for the three Aliens runs, on a machine shared with other runs), **~11 min** on a quiet machine (benchmark). Modern agents do the same budget in **3–11 min**.
2. **The 2018 results reproduce on Aliens and Wafer Thin Mints** (Aliens learning curve 8 / 30 / 57 / 69% wins per 500 games vs the report's 4 / 22 / 58 / 68; Wafer 10.2 / 35% vs 10.2 / 33%; the DQN → DDQN ablation too). Deceptizelda varies between seeds (79%, 38–100, vs 2018's single 99%) and Chase is near zero either way (0.2 vs 0.7). MCTS at 40 ms per move matches 2018 on all four games.
3. **New: on three games, holding one button matches most agents.** Holding "shoot" wins 82% of Aliens games (36.3 points); holding "right" wins every Deceptizelda game; holding any direction wins every Wafer Thin Mints game, by surviving without eating. Scores are trivial too: random play gets 10.8 on Wafer, more than any agent. Chase has no meaningful result at all. **Aliens score is the one clear measure of skill**, and there the 2018 agent's 50 points is well above the baseline (36): the one 2018-budget result that clearly is.
4. **On Aliens' score, today's methods clearly beat 2018 given more experience:** 58 points at 5–10M ticks (62 from one IMPALA seed; two more finish ~09:45) against the 2018 agent's 50 (and 36 for holding shoot). At 2018's budget they roughly match it: Rainbow with a bigger network (IMPALA, 2018) gets 54.7 / 85% vs 50.1 / 81%, within noise, and at 2–3× the wall time per tick; four times more updates per tick gives DQN the same score (53.0).
5. **The 2018 reward's win-rate gains are mostly the trivial strategy.** Its −0.5 penalty on every move of a lost game lifts DQN's Aliens wins from 57% to 85% but not its score (39.4, close to holding shoot's 36.3). PPO, with or without the penalty, learns exactly "hold shoot" on the training levels (identical games to the baseline). On Wafer the penalty drives Rainbow and PPO to the survive-don't-eat strategy (91–97% wins, 2–3 points).
6. **Deceptizelda is not solved by anyone.** Every agent, 2018 and 2026, takes the small exit (2 points: key + door); the human took the large one (10 points). That also explains the 2018 report's "DDQN beat MCTS here": both took the small exit, MCTS less reliably.
7. **Chase stays unsolved**: no agent wins a game, and random play (1.5 points) outscores every agent except Rainbow after 20M ticks (1.7). **MCTS still beats every learning agent on score** (Aliens 77.8 vs ≤62), with no training.

## What was compared

All rows: 50 evaluation games, greedy, levels 0–2 (Deceptizelda: 0 and 2, see caveats). 2026 rows are the mean of 3 seeds; the range is win % across seeds. "2018's ticks" = the game ticks 2018 used for that game.

| Agent | Aliens | Chase | Deceptizelda | Wafer Thin Mints |
|---|---|---|---|---|
| Human (2018) | 42.6 (100%) | 1.2 (0%) | 10.6 (80%) | 11 (0%) |
| One button held all game (best button; new) | 36.3 (82%) shoot | 0.0 (0%) | 2.0 (100%) right | 1.6 (100%) any direction |
| Random play (new) | 33.7 (8%) | 1.5 (0%) | 1.0 (66%) | 10.8 (4%) |
| MCTS, 40 ms per move (2018) | 79.2 (100%) | 3.5 (3.7%) | 1.91 (84.4%) | 9.9 (75.5%) |
| 2018 final agent (2018 report) | 56.3 (82.6%) | 0.7 (0%) | 1.99 (99%) | 10.2 (33%) |
| MCTS, 40 ms per move (2026 PC) | 77.8 (100%) | 3.5 (10%) | 1.9 (90%) | 9.7 (75%) |
| 2018 final agent, replayed | 50.1 (81%, 66–92) | 0.2 (0%) | 1.6 (79%, 38–100) | 10.2 (35%, 32–38) |
| Modern DQN, 2018's ticks | 39.5 (57%, 42–84) | 0.0 (0%) | 1.3 (67%, 0–100) | 7.7 (52%, 34–70) |
| Modern DQN + 2018's reward\*, 2018's ticks | 39.4 (85%, 80–88) | – | – | 7.5 (59%, 32–86) |
| Rainbow + 2018's reward\*, 2018's ticks | 39.6 (88%, 82–94) | 0.0 (0%) | 2.0 (100%) | 2.7 (91%, 76–100) |
| Rainbow, 2018's ticks | 43.2 (66%, 62–72) | 0.0 (0%) | 1.0 (42%, 0–100) | 7.4 (57%, 42–86) |
| Rainbow, IMPALA network, 2018's ticks | 54.7 (85%, 82–90) | – | – | – |
| PPO, 2018's ticks | 37.6 (82%, 78–86)† | 0.0 (0%) | 2.0 (100%) | 6.3 (73%, 72–74) |
| Modern DQN, up to 5M ticks | 58.6 (89%, 84–94) | 0.6 (0%) | 2.0 (100%) | 9.5 (41%, 38–44) |
| Rainbow, up to 5M ticks | 57.0 (93%, 90–98) | 0.0 (0%) | 2.0 (100%) | 9.5 (35%, 20–60) |
| Best 5M setup (by wins, then score), up to 10M ticks (100 games)‡ | 58.0 (95%, 93–98) | 0.8 (0%) | 2.0 (100%) | 10.2 (31%, 24–37) |

\* 2018's reward = ±1 for the game's outcome plus −0.5 on every move of a lost game: outcome shaping that uses knowledge of the game, not part of the standard methods.

† Seed 0 plays the same 50 games as holding shoot, game for game; seeds 1–2 (different evaluation games) match PPO with the 2018 reward game for game, which points to the same fixed policy.

‡ On Wafer neither axis shows skill (random play scores 10.8, holding a direction wins 100%); only MCTS does well on both (9.7 / 75%). Picked from the runs finished before the night (Aliens and Deceptizelda: Rainbow; Chase and Wafer: DQN), so the night's variants, including the IMPALA network, were not candidates.

Noise: with 3 seeds × 50 games, a win rate's standard error is about ±4–5 points; seed-to-seed spread is often wider (the ranges).

## The reward question (Aliens, modern DQN, 2018's ticks, 3 seeds)

| Reward | Score | Wins | Held-out levels |
|---|---|---|---|
| Hold shoot all game (baseline) | 36.3 | 82% | – |
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
| 2018's reward, raw pixels + squared-error loss (as 2018) | 44.4 | 87% | 79% |
| 2018 agent itself | 50.1 | 81% | – |
| ±1 outcome only, 5M ticks | 57.9 | 92% | 72% |
| 2018's reward, 5M ticks | 48.2 | 91% | 23% |

- **The per-move penalty is the part that lifts wins**; the ±1 outcome alone, louder (±5, ±10) or loss-only does not, at this budget. With 5M ticks the plain outcome gets there too (92%).
- **But the lift is in wins, not score.** Every penalty variant scores 38–44, a few points above holding shoot (36.3 / 82%), against 50 for the 2018 agent and 58+ for longer-trained agents. The penalty mainly teaches "don't lose", which on Aliens is close to the trivial strategy. The penalty-size, fading and replay-size differences (75–90% wins) are within noise.
- **Held-out levels: mixed.** The base setup drops to 31% (seeds 2–48%) and to 23% at 5M ticks, but the other penalty variants keep 49–79%, so the drop may be that setup's seeds rather than a property of the penalty.
- **The remaining score gap is strategy.** In won games the 2018 agent averages 54.3 points and DQN + penalty 42.7: the 2018 agent also shoots the protective bases.
- **Wafer Thin Mints is a trade-off by design**: eating scores points and risks death; surviving to tick 1500 wins. With the penalty, Rainbow and PPO learn to survive and eat little (91–97% wins, 2–3 points; holding any direction gets 100% at 1.6); without it, agents eat more (~10 points, ~35–40% wins); DQN with the penalty sits between (7.5 / 59%). Only MCTS gets both (9.7 / 75%).

## Network, algorithm and training budget (Aliens)

| Change | 2018's ticks: score (wins) | Held-out | Up to 5M ticks |
|---|---|---|---|
| Modern DQN | 39.5 (57%) | 61% | 58.6 (89%) |
| DQN, IMPALA network | 52.7 (73%) | 71% | 62.0 (98%), 1 seed\* |
| Rainbow | 43.2 (66%) | 89% | 57.0 (93%) |
| Rainbow, IMPALA network | **54.7 (85%)** | 78% | 57.6 (90%), 1 seed\* |
| Rainbow, IMPALA + 2018's reward | 43.4 (95%) | 69% | – |
| PPO (policy gradient, not Q-learning) | 37.6 (82%) = holding shoot | 72% | 45.9 (83%) |
| DQN, 2× / 4× more updates per tick | 47.2 (35%) / 53.0 (70%) | 57% / 46% | – |
| Rainbow, 2× / 4× more updates per tick | 46.8 (63%) / 51.0 (65%) | 73% / 65% | – |
| DQN / Rainbow trained on all 5 levels | – | – | 55.8 (79%) / 56.0 (81%); "held-out" 83% / 77%, but those levels were seen |
| Rainbow to 10M ticks (stopped at ~7.7M) | – | – | 58.0 (95%), held-out 80% |

\* Seeds 1–2 finish around 09:45; this table will change.

- **At 2018's budget, a bigger network or more updates per tick both add ~11–14 points** over modern DQN (39.5 → 52.7 / 53.0). The IMPALA network costs 2–3× the wall time per tick (9–11 min per 1M-tick run vs 3–5), so at equal compute the comparison is closer than the table suggests. More updates per tick make wins unstable (35–70%).
- **PPO learns "hold shoot"** on Aliens' training levels at 2018's budget (game for game the same as the baseline), and still wins 72% of held-out games. At 5M ticks it scores 45.9, above the baseline. Its 100% on Deceptizelda says little: holding right also wins every game.
- **Beyond ~5M ticks Aliens plateaus** around 57–62 points, still well short of MCTS's 78.
- **Not tested:** PPO with the IMPALA network; equal-compute comparisons; 2018's 4-frame stacks with modern agents.

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
