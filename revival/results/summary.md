# Results

Average score (win %) over 50 evaluation games, greedy, levels 0–2 (Deceptizelda 0 and 2). For 2026 runs: mean over 3 seeds, then the range of win % across seeds; 2018 rows are single runs.

| Agent | Aliens | Chase | Deceptizelda | Wafer Thin Mints |
|---|---|---|---|---|
| Human (2018) | 42.6 (100%) | 1.2 (0%) | 10.6 (80%) | 11 (0%) |
| MCTS, 40 ms (2018) | 79.2 (100%) | 3.5 (3.7%) | 1.91 (84.4%) | 9.9 (75.5%) |
| 2018 final agent (2018 report) | 56.3 (82.6%) | 0.7 (0%) | 1.99 (99%) | 10.2 (33%) |
| MCTS, 40 ms (2026 PC) | 79.1 (100%) | – | – | – |
| 2018 final agent, replayed | 50.1 (81%, 66–92) | 0.2 (0%, 0–0) | 1.6 (79%, 38–100) | 10.2 (35%, 32–38) |
| 2018 final agent, bugs fixed | 48.2 (80%) | – | – | – |
| Modern DQN, 2018's ticks | 39.5 (57%, 42–84) | 0.0 (0%, 0–0) | 1.3 (67%, 0–100) | 7.7 (52%, 34–70) |
| Modern DQN + 2018's reward, 2018's ticks | 39.4 (85%, 80–88) | – | – | 7.5 (59%, 32–86) |
| Rainbow, 2018's ticks | 43.2 (66%, 62–72) | 0.0 (0%, 0–0) | 1.0 (42%, 0–100) | 7.4 (57%, 42–86) |
| Modern DQN, up to 5M ticks | 58.6 (89%, 84–94) | 0.6 (0%, 0–0) | 2.0 (100%, 100–100) | 9.5 (41%, 38–44) |
| Rainbow, up to 5M ticks | 57.0 (93%, 90–98) | 0.0 (0%, 0–0) | 2.0 (100%, 100–100) | 9.5 (35%, 20–60) |

## Held-out levels 3–4 (never trained on)

| Agent | Aliens | Chase | Deceptizelda | Wafer Thin Mints |
|---|---|---|---|---|
| Modern DQN, 2018's ticks | 35.1 (61%, 28–78) | 0.0 (0%, 0–0) | 1.3 (67%, 0–100) | 7.8 (51%, 38–68) |
| Modern DQN + 2018's reward, 2018's ticks | 25.9 (31%, 2–48) | – | – | 7.4 (55%, 20–92) |
| Rainbow, 2018's ticks | 39.1 (89%, 82–100) | 0.0 (0%, 0–0) | 1.0 (46%, 0–100) | 6.8 (59%, 38–86) |
| Modern DQN, up to 5M ticks | 35.8 (69%, 46–80) | 0.1 (0%, 0–0) | 2.0 (100%, 100–100) | 9.7 (37%, 34–42) |
| Rainbow, up to 5M ticks | 37.7 (70%, 64–82) | 0.0 (0%, 0–0) | 2.0 (100%, 100–100) | 9.8 (39%, 28–54) |

## 2018's ablation on Aliens: win % per 500 training games

| Variant | 2018 report | 2026 replay (mean of seeds) | 2026 eval score (wins) |
|---|---|---|---|
| DQN | 6 / 12 / 29 / 35 | 6 / 12 / 26 / 31 | 39.9 (33%, 8–74) |
| DDQN | 3 / 17 / 47 / 57 | 8 / 19 / 44 / 59 | 46.3 (81%, 66–94) |
| Dueling | – (figure only) | 7 / 28 / 61 / 70 | 45.6 (76%, 54–90) |
| DDQN + PER + new reward (final) | 4 / 22 / 58 / 68 | 8 / 30 / 57 / 69 | 50.1 (81%, 66–92) |
| Final, bugs fixed | – (figure only) | 7 / 28 / 53 / 57 | 48.2 (80%) |

## Improvement ladder on Aliens, 2018's ticks (1.01M), seed 0 unless noted

| Step | Eval score (wins) | Train time |
|---|---|---|
| 2018 final agent, replayed | 50.1 (81%, 66–92) | 27.4 min |
| Modern DQN (Double, Huber, Adam, /255 inputs) | 39.5 (57%, 42–84) | 4.0 min |
| + prioritised replay | 40.1 (62%) | 4.0 min |
| + dueling | 43.0 (50%) | 4.2 min |
| + 3-step returns | 46.2 (82%) | 4.3 min |
| + noisy nets + distributional = Rainbow | 43.2 (66%, 62–72) | 2.7 min |

## Training time for one run

| Run | Game ticks | Wall clock | Ticks/s |
|---|---|---|---|
| 2018 plain DQN on Aliens, GTX 980 (report) | ~1.0M | ~12 h | ~23 |
| 2018 final agent replayed, Aliens (mean of 3) | 0.99M | 27 min | 604 |
| Modern DQN, Aliens, 2018's ticks (mean of 3) | 1.01M | 4 min | 4,225 |
| Rainbow, Aliens, 2018's ticks (mean of 3) | 1.01M | 3 min | 6,325 |
| Modern DQN, Aliens, 5M ticks (mean of 3) | 5.00M | 15 min | 5,377 |
| Rainbow, Aliens, 5M ticks (mean of 3) | 5.00M | 14 min | 6,165 |

Long runs stopped early (no longer improving after 3M ticks): chase-dqn-long-s0 at 4.3M, chase-dqn-long-s2 at 5.0M, chase-rainbow-long-s0 at 3.3M, chase-rainbow-long-s1 at 3.5M, chase-rainbow-long-s2 at 4.3M, deceptizelda-dqn-long-s0 at 3.3M, deceptizelda-dqn-long-s1 at 3.3M, deceptizelda-dqn-long-s2 at 3.3M, deceptizelda-rainbow-long-s0 at 3.5M, deceptizelda-rainbow-long-s1 at 3.3M, deceptizelda-rainbow-long-s2 at 3.3M, waferthinmints2018-dqn-long-s0 at 4.5M, waferthinmints2018-dqn-long-s1 at 5.0M, waferthinmints2018-dqn-long-s2 at 3.3M, waferthinmints2018-rainbow-long-s0 at 3.5M, waferthinmints2018-rainbow-long-s1 at 3.3M, waferthinmints2018-rainbow-long-s2 at 3.8M
