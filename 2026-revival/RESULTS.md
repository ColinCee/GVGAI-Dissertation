# GVGAI DQN dissertation, 2018 → 2026: results

What happened when the 2018 dissertation (DQN and its improvements on four GVGAI games) was rebuilt on today's software and hardware, re-run exactly, and then compared with today's methods. Full tables: [results/summary.md](results/summary.md) (regenerate with `uv run python -m tools.report`). Every speed change: [IMPROVEMENTS.md](IMPROVEMENTS.md). Setup and caveats: [README.md](README.md).

## Headlines

The Aliens numbers the blog uses are the overnight check below (5 seeds, one code version, 100 test games each). The sections after it are earlier batches, mostly 3 seeds and 50 games, kept as dated records; where they disagree, the overnight check stands.

1. **Same agent, same 1M game ticks: ~12 h in 2018 → ~25 min today** in practice (23–34 min for the three Aliens runs, on a machine shared with other runs), **~11 min** on a quiet machine (benchmark). Modern agents do the same budget in **3–11 min**.
2. **The 2018 results reproduce on Aliens and Wafer Thin Mints** (Aliens learning curve 8 / 30 / 57 / 69% wins per 500 games vs the report's 4 / 22 / 58 / 68; Wafer 10.2 / 35% vs 10.2 / 33%; the DQN → DDQN ablation too). Deceptizelda varies between seeds (79%, 38–100, vs 2018's single 99%) and Chase is near zero either way (0.2 vs 0.7). MCTS at 40 ms per move matches 2018 on all four games.
3. **New: on three games, holding one button matches most agents.** Holding "shoot" wins 82.4% of Aliens games (37.3 points, on the agents' 500 test games); holding "right" wins every Deceptizelda game; holding any direction wins every Wafer Thin Mints game, by surviving without eating. Scores are trivial too: random play gets 10.8 on Wafer, more than any agent. Chase has no meaningful result at all. **Aliens score is the one clear measure of skill**, and there the 2018 agent's 50 points is well above the baseline (36), as are 2026 agents with a bigger network or more updates per tick (51–55) at the same budget.
4. **On Aliens' score, today's methods clearly beat 2018 given more experience:** 58–62 points at 5–10M ticks with a move every 4 ticks, and 79–83 at ~100% wins with a move every tick or every 2 (best: ±1 outcome + 0.1 × score, IMPALA network, a move every 2 ticks: 82.7 / 99.0% over 5 seeds × 100 games, each step of that chain significant by Welch's t-test), against the 2018 agent's 46.9 / 72.8% over 5 seeds (and 37.3 / 82.4% for holding shoot on the same 500 games). See the overnight check below. At 2018's budget they roughly match it: Rainbow with a bigger network (IMPALA, 2018) gets 54.7 / 85% vs 50.1 / 81%, within noise, and at 2–3× the wall time per tick; four times more updates per tick gives DQN the same score (53.0).
5. **The 2018 reward's win-rate gains are mostly the trivial strategy.** Its −0.5 penalty on every move of a lost game lifts DQN's Aliens wins from 57% to 85% but not its score (39.4, close to holding shoot's 36.3). PPO, with or without the penalty, learns exactly "hold shoot" on the training levels (identical games to the baseline). On Wafer the penalty drives Rainbow and PPO to the survive-don't-eat strategy (91–97% wins, 2–3 points).
6. **Deceptizelda is not solved by anyone.** Every agent, 2018 and 2026, takes the small exit (2 points: key + door); the human took the large one (10 points). That also explains the 2018 report's "DDQN beat MCTS here": both took the small exit, MCTS less reliably.
7. **Chase stays unsolved**: no agent wins a game, and random play (1.5 points) outscores every agent except Rainbow after 20M ticks (1.7). **MCTS (77.8 on Aliens, no training) beats every learning agent that acts every 4 ticks (≤63)**; acting every tick (78.7) or every 2 with IMPALA (82.6) matches or beats it.

## The overnight check (Aliens, 5 seeds, one code version, 100 test games each, 25 Oct 2026)

Every setup the blog's part 2 shows, run again side by side: `experiments/verify.txt`, results `aliens-v-*`. Three at a time on one GPU, so wall times run high.

| Setup | Score (range) | Wins (range) | Held-out wins | Wall time per run |
|---|---|---|---|---|
| 2018 agent, 1M ticks | 46.9 (39.4–51.9) | 72.8% (65–82) | – | 27–29 min |
| DQN, 1M | 39.9 (33.3–45.4) | 51.2% (30–78) | 57% | 2 min |
| Rainbow, 1M | 42.7 (39.4–45.0) | 64.2% (47–82) | 89% | 3 min |
| PPO, 1M | 37.3 (36.4–38.3) | 82.4% (80–85) | 62% | 3 min |
| DQN, 5M (points reward) | 56.3 (51.1–58.5) | 89.0% (85–92) | 74% | 16–18 min |
| Rainbow, 5M | 57.3 (56.3–59.2) | 88.0% (83–94) | 73% | 19–22 min |
| PPO, 5M | 45.2 (39.7–49.2) | 69.4% (54–84) | 84% | 12–17 min |
| DQN, 5M, outcome only | 44.1 (41.0–45.4) | 99.6% (99–100) | 97% | 11–12 min |
| DQN, 5M, 2018 reward | 49.0 (46.5–50.7) | 95.8% (94–98) | 39% | 10–11 min |
| DQN, 5M, outcome + 0.1 × score | 58.9 (57.1–60.2) | 98.8% (97–100) | 93% | 12–13 min |
| … + IMPALA | 62.0 (59.2–64.2) | 100.0% (100–100) | 97% | 25–26 min |
| … + IMPALA + a move every 2 ticks | 82.7 (82.4–83.2) | 99.0% (97–100) | 100% | 43–47 min |
| … + IMPALA + a move every tick | 82.5 (82.2–82.9) | 100% (100–100) | 100% | 113–114 min (seeds 0–2 ran earlier, on a version with the same learning code, paused at times for games) |
| Holding shoot (no learning) | 37.3 (36.4–38.3) | 82.4% (80–85) | – | – |
| DQN, 5M, outcome + 0.05 × score | 55.6 | 99.4% | – | – |
| DQN, 5M, outcome + 0.2 × score | 60.6 | 98.8% | – | – |

- **Nothing moved significantly between the earlier runs and these**: for all 14 setups run both ways, wins and score differ by p > 0.05 (Welch), in both directions, so the 3-seed numbers were noisier, not wrong.
- **Holding shoot and the modern agents share their 500 test games** (engine seed = run seed + 7); the 2018 agent is tested on other games drawn from the same levels, so its comparisons are unpaired.
- **The 2018 agent is noisier than 3 seeds showed**: these 5 runs average 72.8% wins (65–82) and 46.9 points, against 81.3% and 50.1 from the first 3; its 8 runs together span 65–92%. Holding shoot (82%) now sits at the top of its range.
- **The bigger network's gain holds up**: 62.0 points (59.2–64.2) against 58.9 (57.1–60.2), Welch p = 0.025, so the greedy chain (AGENTS.md) keeps it. Every other step decides the same way as before: more practice p = 0.013 (wins), the outcome + score reward p = 0.001 (wins), a move every 2 ticks p < 0.001 (score); DQN vs Rainbow at 5M (p ≈ 0.7) is a tie.
- **A move every tick + IMPALA ties a move every 2 ticks** at 5 seeds (wins p = 0.19, score p = 0.24), at about 2½× the training time, so the chain keeps every 2 ticks.
- **PPO's wins don't rise with 5M ticks** (82.4% → 69.4%, 54–84, Welch p = 0.085, a tie), though its score does (37.3 → 45.2). The best setup loses 5 games in 500 (0, 2, 0, 3, 0). PPO at 1M plays exactly like holding shoot: the same score in every one of the 500 test games.

## The Rainbow branch (Aliens, 5 seeds, 100 test games each, 25 Oct 2026)

The same chain with Rainbow in place of DQN: `experiments/verify-rainbow.txt`, results `aliens-v-rainbow-*` (seed 0 of the last row rerun after a compile-cache crash).

| Setup | DQN score / wins | Rainbow score / wins | Welch p (wins, score) |
|---|---|---|---|
| 5M, outcome + 0.1 × score | 58.9 / 98.8% | 50.1 / 98.4% | 0.59, < 0.001 |
| … + IMPALA | 62.0 / 100.0% | 60.0 / 97.8% | 0.004, 0.11 |
| … + IMPALA + a move every 2 ticks | 82.7 / 99.0% | 75.7 / 98.8% | 0.79, 0.036 |

- **Rainbow never beats DQN on this chain**: it ties or loses on wins and scores lower at every step, so keeping the simpler DQN at step 1 cost nothing.

## Prioritised replay (Aliens, 5 seeds, 100 test games each, 26 Oct 2026)

The 2018 agent's other trick, put back: `experiments/verify-per.txt` (step 1 settings) and `verify-per-final.txt` (the final agent), results `aliens-v-dqn-per-5m-*` and `aliens-v-per-score01-ar2-impala-*`. Same seeds, test games and learning code as the overnight check, a day later.

| Setup | Without PER (score / wins) | With PER | Welch p (wins, score) |
|---|---|---|---|
| Double DQN, 5M, points reward | 56.3 / 89.0% | 58.9 / 93.0% | 0.048, 0.14 |
| The final agent (IMPALA, outcome + 0.1 × score, a move every 2 ticks) | 82.7 / 99.0% | 80.8 / 99.6% | 0.42, 0.007 |

- **Step 1: a tie after Holm.** Against the step's three other options, the 0.048 needs p < 0.025, so the simpler Double DQN stays.
- **The final agent: worse.** Wins tie, score drops 1.9 (p = 0.007), and runs take 63 min instead of 44. Dropping PER was right for this chain.

## Jev, a decision model, with no training (Aliens, 26 Oct 2026)

[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) (TypeSafe, `jev-latest`, which reported itself as `jev-1.14.0`) gets one API call per move: the game as text from the engine's sprite positions, one question, and the choices shoot / left / right, each with a probability back. The player plays the most likely one, a move every 2 ticks. It's `baselines/jev.py`, and every game's result and token count is in `results/jev/`. The games are the agents' own test games: block *k* is the 8 streams × 13 games that `agents/evaluate.py` plays for `--seed k`.

| Run | Games | Won | Points | Tokens a call | Cost |
|---|---|---|---|---|---|
| Text state, block 0, streams 0–1 (`jev-latest-s0`) | 26 | 15 (57.7%) | 43.0 | 555 | $0.15 |
| + a line naming the bombs within 2 columns of the ship (`--warn`, block 0) | 104 | 83 (79.8%) | 48.2 | 564 | $0.68 |
| Same, state as a JSON object (`--warn --json --v1`, block 0) | 104 | 78 (75.0%) | 41.3 | 604 | $0.73 |
| `jev-preview`, `--warn`, block 1, shield hits described correctly | 104 | 80 (76.9%) | 44.0 | 583 | $0.63 |
| Holding shoot (saved baseline, 5 blocks of 100) | 500 | 82.4% | 37.3 | | free |
| Rule: shoot, step away from a bomb within 2 columns and 3 rows above (`--rule`, 5 blocks) | 520 | 99.2% (96–100) | 42.2 | | free |

- **Jev wins about as often as holding shoot and scores more.** One block per setup, so the win rates are ±8 points. Its points are above holding shoot's range on all 5 blocks (36.4–38.3).
- **The `--warn` line was chosen after seeing streams 0–1**: on those 26 games it won 22 against 15 without it. On the 78 games after that it won 61.
- **JSON state is no better:** the win gap could be luck (z = 0.8), and points are lower (Welch t = 4.0 per game).
- **The rule uses the same positions and wins far more,** so the description does most of the work, not the model. Jev's 46% average for the move it picks also shows it's rarely sure.
- **Cost and speed:** $2.19 for the 338 scored games, and about $2.23 in total once the test games and the recorded clips are added. That's about $0.006 a game, and moves take 180–200 ms at the median.
- **Caveats:** the engine doesn't replay games exactly (holding shoot twice on block 0's seeds won 93 and then 89 of 104), so the comparisons are by block, not game by game. Block 0's question said shield hits were "wasted", but they score 1 point. That's fixed for the preview run (`QUESTION` against `QUESTION_V1`).

## Why a trained agent outscores MCTS (Aliens, 51 games, 200 searches per move, 26 Oct 2026)

MCTS (the engine's sampleMCTS) scores the same at 50, 200 or ~340 searches per move (77.8–77.9), so search budget isn't its limit. Its rollouts are random moves; looking further ahead makes it worse, not better:

| Rollout depth | Score | Wins |
|---|---|---|
| 10 (default) | 77.0 | 100% |
| 30 | 71.6 | 100% |
| 60 | 62.9 | 100% |

Each step down is significant (Welch p < 0.001). A random rollout is a noisy judge of a move, and a longer one adds more noise than signal; the final agent's learned values (looking ~100 decisions ahead) judge moves better, hence 82.7. Run with `baselines/mcts.py --iters 200 --depth N`.

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

‡ On Wafer neither axis shows skill (random play scores 10.8, holding a direction wins 100%); only MCTS does well on both (9.7 / 75%). Picked from the runs finished before the night (Aliens and Deceptizelda: Rainbow; Chase and Wafer: DQN), so the night's variants, including the IMPALA network (DQN + IMPALA reached 62.5 / 96% at 5M, the best Aliens result), were not candidates.

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

## Choosing the reward (Aliens, 5M ticks, 3 seeds)

| Reward | DQN, standard network | Unseen levels | PPO | Unseen levels |
|---|---|---|---|---|
| Score changes only (standard) | 58.6 (89%) | 69% | 45.9 (83%) | 82% |
| Score + ±1 outcome | 57.9 (92%) | 72% | – | – |
| 2018's reward (outcome + −0.5 penalty) | 48.2 (91%) | 23% | – | – |
| **±1 outcome only** (no points) | 44.5 (**100%**, every seed) | **100%** | 43.5 (94%) | 84% |
| ±1 outcome only, 3-step returns | 44.6 (98%) | 72% | – | – |
| **±1 outcome + 0.1 × score changes** | **60.1 (98%)** | 94% | 48.8 (90%) | 84% |
| −1 loss; a win 1 − 0.5c/(c + game score), c = random play (33.7) | 43.5 (99%) | 88% | – | – |

- **Reward the outcome.** With the outcome as the only reward, DQN wins every evaluation game on every seed, on the training levels and on unseen ones. Adding a tenth of the score as a tie-break keeps 98% wins and gives the highest score of any agent on the standard network (60.1). This beats every shaped reward tried, including 2018's, and needs no knowledge of the game.
- **It needs the training.** At 2018's budget the plain outcome was the worst reward (45% wins): one signal at the end of a ~150-move game takes millions of ticks to learn from. That is why 2018's per-move penalty helped then, and why it isn't needed now.
- **Fading the signal with distance from the loss** is what the discount already does; passing it back faster with 3-step returns didn't help here (98% vs 100%, held-out 72%).
- **Paying the score only at the end of a won game doesn't teach scoring** (saturating win reward, 23 Oct 2026): it plays like outcome only (43.5 vs 44.5 points, 99% wins). A 40- and a 60-point win differ by 0.05, which reaches the moves that earned the points only after ~150 steps of discounting; 0.1 × score pays each point when it happens.
- **Recommended reward:** ±1 outcome + 0.1 × score (the setup built on it is in the overnight check). At 5 seeds, 0.2 × score scores 1.6 more (p = 0.047), but with 0.05 and 0.2 both tested against 0.1, Holm needs p < 0.025, so it's a tie and 0.1 stays.

The same two rewards on the other games (DQN, 5M ticks, 3 seeds):

| Game | Outcome only | Outcome + 0.1 × score | Score only | Trivial baseline | MCTS |
|---|---|---|---|---|---|
| Wafer Thin Mints | 4.2 (90%) | 7.8 (67%) | 9.5 (41%) | 1.6 (100%), any direction | 9.7 (75%) |
| Deceptizelda | 2.0 (100%) | 2.0 (100%) | 2.0 (100%) | 2.0 (100%), hold right | 1.9 (90%) |
| Chase | 0.0 (0%) | 0.0 (0%) | 0.6 (0%) | 0.0 (0%) | 3.5 (10%) |

- **Wafer: the weight on score is a dial between surviving and eating.** Outcome only survives (90%); a tenth of the score trades wins for points (7.8 / 67%), the closest any learning agent gets to MCTS on both.
- **Deceptizelda: still the small exit, every game.** A tie-break of 0.1 × 10 = 1 point for the big exit doesn't make the agent explore far enough to find it: an exploration problem, not a reward one.
- **Chase: no reward helps.** No agent ever wins, so the outcome reward never fires.

## Winning and scoring together (Aliens, 5M ticks, 3 seeds, 100 test games each, 23–24 Oct 2026)

All rewarded ±1 for the outcome plus a weight × score changes; one thing changed per row.

| Change from the base (0.1 × score, a move every 4 ticks, standard network) | Score | Wins | Held-out wins | Wall time per run |
|---|---|---|---|---|
| Base | 59.0 | 98.7% | 87% | 14 min |
| Score weight 0.05 | 56.5 | 100% | 86% | 7–12 min |
| Score weight 0.2 | 59.6 | 97.7% | 89% | 11–13 min |
| 10-step returns | 45.6 | 83.0% | 55% | 13–19 min |
| 20-step returns | 42.8 | 85.7% | 57% | 17 min |
| IMPALA network | 62.9 | 100% | 99% | 23–29 min |
| A move every 2 ticks | 76.4 | 95.7% | 97% | 23 min |
| A move every 2 ticks, 0.05 × score | 68.9 | 94.3% | 96% | 23–30 min |
| **A move every tick** | **78.7** | **99.7%** | **100%** | 46–59 min |
| **A move every 2 ticks + IMPALA** | **82.6** | **99.7%** | **100%** | 52–54 min |
| A move every tick + IMPALA | 82.3 | 100% | 100% | ~3 h (paused at times for games) |
| MCTS, no training (reference) | 77.8 | 100% | – | – |

- **How often the agent acts was the missing piece.** Holding each move for 4 ticks capped the score near 60; acting every tick or every 2 matches or beats MCTS (77.8) on the same 5M ticks, the first learning agents to do so. Colin's guess: some positions and shots can't be reached when every move lasts 4 ticks.
- **The best setup** (a move every 2 ticks + IMPALA) beats MCTS on score; it costs about 3.5× the wall time of the base. At 5 seeds (the overnight check) it is 82.7 / 99.0%, and a move every tick + IMPALA ties it at about 2½× its time, so the greedy chain (AGENTS.md) keeps every 2 ticks.
- **Longer n-step returns hurt** (83–86% wins): with an outcome reward they add variance rather than speed.
- **The score weight doesn't set the loss rate**: at 5 seeds (the overnight check) 0.05, 0.1 and 0.2 lose 3, 6 and 6 games in 500, so the earlier 0 / 4 / 7 in 300 (3 seeds) was noise. For score, 0.2 is 1.6 above 0.1 (p = 0.047) and 0.05 is 3.3 below it (p = 0.07). Losses replayed with `uv run python -m tools.losses <run> [--heldout]` on the earlier runs were all deaths mid-game (a bomb onto a ship sitting still and shooting), never the 2000-tick timeout; on the overnight runs a replay is a fresh play of the same games (play isn't bit-for-bit repeatable, LEARNINGS.md), and replaying the best setup's 5 runs lost 2 of 500 (its evaluation lost 5), both deaths mid-game with the ship standing still under a falling bomb.

## Network, algorithm and training budget (Aliens)

| Change | 2018's ticks: score (wins) | Held-out | Up to 5M ticks |
|---|---|---|---|
| Modern DQN | 39.5 (57%) | 61% | 58.6 (89%) |
| DQN, IMPALA network | 52.7 (73%) | 71% | **62.5 (96%)** |
| Rainbow | 43.2 (66%) | 89% | 57.0 (93%) |
| Rainbow, IMPALA network | **54.7 (85%)** | 78% | 58.7 (93%) |
| Rainbow, IMPALA + 2018's reward | 43.4 (95%) | 69% | – |
| PPO (policy gradient, not Q-learning) | 37.6 (82%) = holding shoot | 72% | 45.9 (83%) |
| PPO, reward only ±1 at the game's end (no score) | 39.4 (78%) | 89% | 43.5 (**94%**, 88–100), held-out 84% |
| DQN, 2× / 4× more updates per tick | 47.2 (35%) / 53.0 (70%) | 57% / 46% | – |
| Rainbow, 2× / 4× more updates per tick | 46.8 (63%) / 51.0 (65%) | 73% / 65% | – |
| DQN / Rainbow trained on all 5 levels | – | – | 55.8 (79%) / 56.0 (81%); "held-out" 83% / 77%, but those levels were seen |
| Rainbow to 10M ticks (stopped at ~7.7M) | – | – | 58.0 (95%), held-out 80% |

- **At 2018's budget, a bigger network or more updates per tick both add ~11–14 points** over modern DQN (39.5 → 52.7 / 53.0). The IMPALA network costs 2–3× the wall time per tick (9–11 min per 1M-tick run vs 3–5), so at equal compute the comparison is closer than the table suggests. More updates per tick make wins unstable (35–70%).
- **PPO learns "hold shoot"** on Aliens' training levels at 2018's budget (game for game the same as the baseline), and still wins 72% of held-out games. At 5M ticks it scores 45.9, above the baseline. Its 100% on Deceptizelda says little: holding right also wins every game.
- **PPO rewarded only for the outcome** (+1 win, −1 loss, nothing for points) does not settle on holding shoot, and at 5M ticks wins 94% of games, clearly above the baseline (82%) and level with the best DQN (96%), at a lower score (43.5 vs 62.5): it plays to win, not to score.
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

2018 recorded only one number, "approximately 12 hours" for a 2,000-episode DQN run, on a different PC (i7-2700K, GTX 980, 16 GB DDR3), with no split between the game and the learning. Everything below was measured on the 2026 machine ([LEARNINGS.md](LEARNINGS.md), "Environment"), so a later revisit can rerun the same commands and compare like with like.

| | 2018 | 2026 |
|---|---|---|
| One game, engine alone, ticks/s | ≤40 (a 25 ms pause every tick was built in; 34 measured on the 2026 PC) | ~2,600 |
| Training, ticks/s | ~23 (2018's 2,000 DQN games, 1.18M ticks, in ~12 h: ~10 h per 1M ticks) | 2018 agent ~600–1,500 per run; modern agents ~9,000 per run, ~16,000 with 3 at once |
| One 1M-tick run | ~12 h | 3–11 min |
| This whole project's training (241 runs, 554M ticks) | ~280 days of non-stop training | ~15 h of wall clock (39 h of run time, 3 runs at once) |

**To re-measure** (quiet machine, nothing else running):

| What | Command | October 2026 |
|---|---|---|
| One game, today's engine (random play, 4-tick repeat) | `uv run python bench/game_speed.py` | 2,590 ticks/s |
| 2018 agent, a whole training run | `uv run python bench/train2018_speed.py cuda` / `cpu` | 1,550 / 821 ticks/s: 11 / 20 min per 1M ticks |
| Of that 11 min | game alone (1M ÷ 2,590) vs the rest | 6.4 min game, 4.3 min agent |
| Modern DQN update | `uv run python bench/update_speed.py` | 0.60 ms (standard network) |

Each engine step from 34 to 2,590 ticks/s, with its benchmark and commit tag, is a row in [IMPROVEMENTS.md](IMPROVEMENTS.md).

**Per training setup**, Aliens, 5M ticks, 3 runs at once on the one GPU (results/*-e100-*.json):

| Setup (±1 outcome + 0.1 × score) | Decisions | Ticks/s per run | Wall time per run | Score / wins |
|---|---|---|---|---|
| A move every 4 ticks, standard network | 1.25M | ~6,100 | 14 min | 59.0 / 98.7% |
| A move every 4 ticks, IMPALA | 1.25M | ~2,900–3,600 | 23–29 min | 62.9 / 100% |
| A move every 2 ticks, standard network | 2.5M | ~3,600 | 23 min | 76.4 / 95.7% |
| A move every tick, standard network | 5M | ~1,400–1,800 | 46–59 min | 78.7 / 99.7% |
| A move every 2 ticks, IMPALA | 2.5M | ~1,550–1,600 | 52–54 min | 82.6 / 99.7% |

## Caveats

- One seed per agent in 2018 versus 3 here; 2018's own three runs of the final agent differ by up to 14 win-% points, and a 50-game evaluation is ±10 points. The report's Table 6.1 win rates are training-time figures (Aliens: the last 500 of 2,000 training games of one run, still exploring at ε = 0.1), not a separate test.
- Deceptizelda's level 1 is a bigger map than 2018's network could have played; all agents use levels 0 and 2.
- Chase's only 2018 run used the old reward (no penalty); the replay does too.
- Wafer Thin Mints: the engine's current rules differ from 2018's; the 2018 rules were rebuilt from the engine's history.
- Modern agents see 2×2-averaged greyscale frames; the 2018 replay keeps 2018's PIL Lanczos frames.
- "Lost" for the −0.5 penalty is the game's own verdict, as in 2018's code: on Wafer Thin Mints surviving to tick 1500 is a win, dying a loss.
- MCTS at 40 ms on today's CPU gets more search per move than in 2018 (140–600 iterations), yet scores the same: the games need little search (table at the end).
- The 2018 agent's timing averages one CPU run and two GPU runs.

## MCTS at a fixed search budget

2026's 40 ms per move buys 140–600 iterations; a fixed number of iterations is comparable across machines. 50 games each:

| Iterations per move | Aliens | Chase | Deceptizelda | Wafer Thin Mints |
|---|---|---|---|---|
| 50 | 77.9 (100%) | 3.9 (16%) | 2.0 (84%) | 10.3 (55%) |
| 200 | 77.8 (100%) | 3.5 (6%) | 2.0 (96%) | 9.7 (73%) |
| 40 ms (140–600) | 77.8 (100%) | 3.5 (10%) | 1.9 (90%) | 9.7 (75%) |

A little search is enough on these games: more mostly buys Wafer wins (surviving) at the cost of a little score.
