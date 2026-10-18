# Results

Average score (win %) over 50 evaluation games, levels 0–2. For our runs: mean over seeds, win-% range, seed count.

| Agent | Aliens | Chase | Deceptizelda | Wafer Thin Mints |
|---|---|---|---|---|
| Human (2018) | 42.6 (100%) | 1.2 (0%) | 10.6 (80%) | 11 (0%) |
| MCTS (2018) | 79.2 (100%) | 3.5 (3.7%) | 1.91 (84.4%) | 9.9 (75.5%) |
| DDQN+PER (2018) | 56.3 (82.6%) | 0.7 (0%) | 1.99 (99%) | 10.2 (33%) |
| ddqn2018 (2026) | 51.8 (92%) | – | – | – |

## Aliens learning: win % per 500 training games

| Run | 1–500 | 501–1000 | 1001–1500 | 1501–2000 |
|---|---|---|---|---|
| 2018 run A (report) | 9% | 34% | 61% | 72% |
| 2018 run B (CSV) | 4% | 22% | 71% | 82% |
| aliens-ddqn2018-s0 | 8% | 29% | 61% | 75% |

## Training time

| Run | Game ticks | Wall clock | Ticks/s |
|---|---|---|---|
| 2018 plain DQN, GTX 980 (report) | ~1.0M | ~12 h | ~23 |
| aliens-ddqn2018-s0 | 0.99M | 34 min | 491 |
