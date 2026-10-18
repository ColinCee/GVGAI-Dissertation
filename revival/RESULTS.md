# GVGAI DQN dissertation, 2018 → 2026: results

What happened when the 2018 dissertation (DQN and its improvements on four GVGAI games) was rebuilt on today's software and hardware, re-run exactly, and then compared with today's methods. Full tables: [results/summary.md](results/summary.md) (regenerate with `uv run python report.py`). Every speed change: [IMPROVEMENTS.md](IMPROVEMENTS.md). Setup and caveats: [README.md](README.md).

## Headlines

1. **Same agent, same 1M game ticks: ~12 h in 2018 → ~25 min today** in practice (23–34 min for the three Aliens runs, on a machine shared with other runs), **~11 min** on a quiet machine (benchmark). Modern agents do the same budget in **3–4 min**.
2. **The 2018 results reproduce on Aliens and Wafer Thin Mints.** The replayed final agent lands on the report's numbers (Aliens learning curve 8 / 30 / 57 / 69% wins per 500 games vs the report's 4 / 22 / 58 / 68; Wafer Thin Mints 10.2 / 35% vs 10.2 / 33%), and so does the DQN → DDQN ablation. MCTS on Aliens: 79.1 / 100% vs 79.2 / 100% in 2018. Chase is near zero either way (0.2 vs 0.7); Deceptizelda varies a lot between seeds (79% wins, 38–100) against 2018's single 99% run.
3. **At 2018's budget, the 2018 agent still beats today's methods on Aliens** (50.1 score / 81% wins vs modern DQN 39.5 / 57%, Rainbow 43.2 / 66%). The reason is its **reward**: the −0.5 penalty on every move of a lost game. Give modern DQN the same reward and its win rate jumps to **85%**, steadier across seeds than the 2018 agent. Cleaner-looking alternatives (a ±1 outcome at the end only, or with a longer look-ahead) don't do it (45% and 32%).
4. **With more experience, today's methods pass it:** at 5M ticks modern DQN reaches 58.6 / 89% and Rainbow 57.0 / 93% on Aliens, without the 2018 reward, in ~15 min.
5. **Chase stays unsolved by every agent** (0% wins, as in 2018); **Deceptizelda is solved** (100% wins at 5M ticks).

## What was compared

All rows: 50 evaluation games, greedy, levels 0–2 (Deceptizelda: 0 and 2, see caveats). 2026 rows are the mean of 3 seeds; the range is win % across seeds. "2018's ticks" = the game ticks 2018 used for that game.

| Agent | Aliens | Chase | Deceptizelda | Wafer Thin Mints |
|---|---|---|---|---|
| Human (2018) | 42.6 (100%) | 1.2 (0%) | 10.6 (80%) | 11 (0%) |
| MCTS, 40 ms per move (2018) | 79.2 (100%) | 3.5 (3.7%) | 1.91 (84.4%) | 9.9 (75.5%) |
| 2018 final agent (2018 report) | 56.3 (82.6%) | 0.7 (0%) | 1.99 (99%) | 10.2 (33%) |
| MCTS, 40 ms per move (2026 PC) | 79.1 (100%) | *running* | *running* | *running* |
| 2018 final agent, replayed | 50.1 (81%, 66–92) | 0.2 (0%) | 1.6 (79%, 38–100) | 10.2 (35%, 32–38) |
| Modern DQN, 2018's ticks | 39.5 (57%, 42–84) | 0.0 (0%) | 1.3 (67%, 0–100) | 7.7 (52%, 34–70) |
| Modern DQN + 2018's reward\*, 2018's ticks | 39.4 (85%, 80–88) | *queued* | *queued* | 7.5 (59%, 32–86) |
| Rainbow, 2018's ticks | 43.2 (66%, 62–72) | 0.0 (0%) | 1.0 (42%, 0–100) | 7.4 (57%, 42–86) |
| Modern DQN, up to 5M ticks | 58.6 (89%, 84–94) | 0.6 (0%) | 2.0 (100%) | 9.5 (41%, 38–44) |
| Rainbow, up to 5M ticks | 57.0 (93%, 90–98) | 0.0 (0%) | 2.0 (100%) | 9.5 (35%, 20–60) |

\* 2018's reward = ±1 for the game's outcome plus −0.5 on every move of a lost game: outcome shaping that uses knowledge of the game, not part of the standard methods.

## The reward question (Aliens, modern DQN, 2018's ticks, 3 seeds)

| Reward | Score | Wins |
|---|---|---|
| Score changes only (standard) | 39.5 | 57% |
| + ±1 outcome at game end | 40.5 | 45% |
| + ±1 outcome, longer look-ahead (γ 0.997) | 38.1 | 32% |
| **+ ±1 outcome + 2018's −0.5 on every move of a lost game** | 39.4 | **85%** |
| 2018 agent itself | 50.1 | 81% |

Why the penalty works and the clean outcome signal doesn't is still open. One idea: a ±1 at the end of a ~150-move game fades to ~0.2 by the early moves (0.99¹⁵⁰) and competes with dozens of +1s for kills, while the penalty puts the signal on every move. But a longer look-ahead, which should help if that were the whole story, made it worse (32%), so the overnight runs test louder outcomes (±5, ±10), loss-only, a penalty that fades with distance from the loss, and the outcome signal over 5M ticks. The remaining score gap looks like strategy rather than skill: in won evaluation games (over 150 per agent) the 2018 agent averages 54.3 points and modern DQN + penalty 42.7. On Aliens, extra points come from shooting the protective bases, the "deceptive" points the 2018 report described.

One warning sign: on held-out levels (3–4, never trained on; [table](results/summary.md#held-out-levels-34-never-trained-on)) modern DQN + 2018's reward drops to 31% wins on Aliens, while Rainbow at the same budget keeps 89%. The shaped reward may fit the training levels too closely.

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
| This whole project's training so far (101 runs, 208M ticks) | ~105 days of non-stop training | ~6 h of wall clock (18 h of run time, several runs at once) |

## Caveats

- One seed per agent in 2018 versus 3 here; 2018's own two runs of the same agent differ by 10–17 win-% points, and a 50-game evaluation is ±10 points.
- Deceptizelda's level 1 is a bigger map than 2018's network could have played; all agents use levels 0 and 2.
- Chase's only 2018 run used the old reward (no penalty); the replay does too.
- Wafer Thin Mints: the engine's current rules differ from 2018's; the 2018 rules were rebuilt from the engine's history.
- Modern agents see 2×2-averaged greyscale frames; the 2018 replay keeps 2018's PIL Lanczos frames.
- "Lost" for the −0.5 penalty is the game's own verdict, as in 2018's code: on Wafer Thin Mints surviving to tick 1500 is a win, dying a loss.
- MCTS at 40 ms on today's CPU gets far more search per move than in 2018; a fixed-search-budget MCTS row (comparable across machines) is still to do.
- The 2018 agent's timing averages one CPU run and two GPU runs.

## Still running

MCTS on Chase, Deceptizelda and Wafer; Rainbow + 2018's reward on all games; modern DQN + 2018's reward to 5M ticks; penalty size and penalty-only; two checks on the remaining score gap (2018's replay size; raw pixels + MSE as 2018). Then more reward styles (louder outcome ±5/±10, loss-only, a penalty that fades with distance from the loss, outcome-only for longer) and the best combination for top performance.
