# Results

Average score (win %) over 50 evaluation games, greedy, levels 0–2 (Deceptizelda 0 and 2). For 2026 runs: mean over 3 seeds, then the range of win % across seeds; 2018 rows are single runs.

| Agent | Aliens | Chase | Deceptizelda | Wafer Thin Mints |
|---|---|---|---|---|
| Human (2018) | 42.6 (100%) | 1.2 (0%) | 10.6 (80%) | 11 (0%) |
| MCTS, 40 ms (2018) | 79.2 (100%) | 3.5 (3.7%) | 1.91 (84.4%) | 9.9 (75.5%) |
| 2018 final agent (2018 report) | 56.3 (82.6%) | 0.7 (0%) | 1.99 (99%) | 10.2 (33%) |
| One button held all game (best button) | 36.3 (82%) use | 0.0 (0%) down | 2.0 (100%) right | 1.6 (100%) right |
| Random play | 33.7 (8%) | 1.5 (0%) | 1.0 (66%) | 10.8 (4%) |
| MCTS, 40 ms (2026 PC) | 77.8 (100%) | 3.5 (10%) | 1.9 (90%) | 9.7 (75%) |
| 2018 final agent, replayed | 50.1 (81%, 66–92) | 0.2 (0%, 0–0) | 1.6 (79%, 38–100) | 10.2 (35%, 32–38) |
| 2018 final agent, bugs fixed | 48.2 (80%) | – | – | – |
| Modern DQN, 2018's ticks | 39.5 (57%, 42–84) | 0.0 (0%, 0–0) | 1.3 (67%, 0–100) | 7.7 (52%, 34–70) |
| Modern DQN + 2018's reward, 2018's ticks | 39.4 (85%, 80–88) | – | – | 7.5 (59%, 32–86) |
| Rainbow, 2018's ticks | 43.2 (66%, 62–72) | 0.0 (0%, 0–0) | 1.0 (42%, 0–100) | 7.4 (57%, 42–86) |
| Rainbow, IMPALA network, 2018's ticks | 54.7 (85%, 82–90) | – | – | – |
| PPO, 2018's ticks | 37.6 (82%, 78–86) | 0.0 (0%, 0–0) | 2.0 (100%, 100–100) | 6.3 (73%, 72–74) |
| Modern DQN, up to 5M ticks | 58.6 (89%, 84–94) | 0.6 (0%, 0–0) | 2.0 (100%, 100–100) | 9.5 (41%, 38–44) |
| Rainbow, up to 5M ticks | 57.0 (93%, 90–98) | 0.0 (0%, 0–0) | 2.0 (100%, 100–100) | 9.5 (35%, 20–60) |
| Best 5M setup, up to 10M ticks, 100 eval games | 58.0 (95%, 93–98) | 0.8 (0%, 0–0) | 2.0 (100%, 100–100) | 10.2 (31%, 24–37) |

## Held-out levels 3–4 (never trained on)

| Agent | Aliens | Chase | Deceptizelda | Wafer Thin Mints |
|---|---|---|---|---|
| Modern DQN, 2018's ticks | 35.1 (61%, 28–78) | 0.0 (0%, 0–0) | 1.3 (67%, 0–100) | 7.8 (51%, 38–68) |
| Modern DQN + 2018's reward, 2018's ticks | 25.9 (31%, 2–48) | – | – | 7.4 (55%, 20–92) |
| Rainbow, 2018's ticks | 39.1 (89%, 82–100) | 0.0 (0%, 0–0) | 1.0 (46%, 0–100) | 6.8 (59%, 38–86) |
| Rainbow, IMPALA network, 2018's ticks | 40.7 (78%, 70–94) | – | – | – |
| PPO, 2018's ticks | 35.0 (72%, 64–80) | 0.0 (0%, 0–0) | 2.0 (100%, 100–100) | 5.8 (77%, 66–86) |
| Modern DQN, up to 5M ticks | 35.8 (69%, 46–80) | 0.1 (0%, 0–0) | 2.0 (100%, 100–100) | 9.7 (37%, 34–42) |
| Rainbow, up to 5M ticks | 37.7 (70%, 64–82) | 0.0 (0%, 0–0) | 2.0 (100%, 100–100) | 9.8 (39%, 28–54) |
| Best 5M setup, up to 10M ticks, 100 eval games | 39.9 (80%, 71–93) | 0.0 (0%, 0–0) | 2.0 (100%, 100–100) | 10.2 (32%, 21–43) |

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

## Every run group

| Runs | Seeds | Eval score (wins) | Held-out levels | Ticks | Wall clock |
|---|---|---|---|---|---|
| aliens-best | 3 | 58.0 (95%, 93–98) | 39.9 (80%, 71–93) | 7.67M | 40 min |
| aliens-ddqn2018 | 3 | 50.1 (81%, 66–92) | – | 0.99M | 27 min |
| aliens-ddqn2018-ddqn | 3 | 46.3 (81%, 66–94) | – | 1.14M | 24 min |
| aliens-ddqn2018-dqn | 3 | 39.9 (33%, 8–74) | – | 1.17M | 23 min |
| aliens-ddqn2018-dueling | 3 | 45.6 (76%, 54–90) | – | 1.04M | 21 min |
| aliens-ddqn2018fixed | 1 | 48.2 (80%) | – | 1.02M | 19 min |
| aliens-dqn-alllevels-long | 3 | 55.8 (79%, 74–84) | 45.2 (83%, 64–96) | 5.00M | 13 min |
| aliens-dqn-b18 | 3 | 39.5 (57%, 42–84) | 35.1 (61%, 28–78) | 1.01M | 4 min |
| aliens-dqn-impala-b18 | 3 | 52.7 (73%, 58–86) | 37.6 (71%, 58–78) | 1.01M | 9 min |
| aliens-dqn-impala-long | 3 | 62.5 (96%, 94–98) | 38.7 (80%, 76–84) | 4.42M | 47 min |
| aliens-dqn-long | 3 | 58.6 (89%, 84–94) | 35.8 (69%, 46–80) | 5.00M | 15 min |
| aliens-dqn-loss-only-b18 | 3 | 40.6 (44%, 14–64) | 35.1 (57%, 38–76) | 1.01M | 2 min |
| aliens-dqn-outcome-b18 | 3 | 40.5 (45%, 18–66) | 25.8 (31%, 24–38) | 1.01M | 2 min |
| aliens-dqn-outcome-g997-b18 | 3 | 38.1 (32%, 16–62) | 28.7 (29%, 0–84) | 1.01M | 2 min |
| aliens-dqn-outcome-long | 3 | 57.9 (92%, 90–94) | 35.4 (72%, 62–84) | 5.00M | 12 min |
| aliens-dqn-outcome10-b18 | 3 | 40.7 (61%, 46–70) | 28.1 (36%, 16–54) | 1.01M | 3 min |
| aliens-dqn-outcome5-b18 | 3 | 36.5 (54%, 30–68) | 34.1 (54%, 34–64) | 1.01M | 3 min |
| aliens-dqn-penalty-decay-b18 | 3 | 41.6 (75%, 70–80) | 25.0 (41%, 12–62) | 1.01M | 2 min |
| aliens-dqn-penalty-only-b18 | 3 | 38.5 (81%, 78–86) | 32.1 (49%, 38–62) | 1.01M | 2 min |
| aliens-dqn-per | 1 | 40.1 (62%) | 26.9 (46%) | 1.01M | 4 min |
| aliens-dqn-per-dueling | 1 | 43.0 (50%) | 32.8 (48%) | 1.01M | 4 min |
| aliens-dqn-per-dueling-nstep | 1 | 46.2 (82%) | 29.8 (32%) | 1.01M | 4 min |
| aliens-dqn-reward2018-b18 | 3 | 39.4 (85%, 80–88) | 25.9 (31%, 2–48) | 1.01M | 2 min |
| aliens-dqn-reward2018-buf100k-b18 | 3 | 42.3 (88%, 84–92) | 37.1 (78%, 66–94) | 1.01M | 2 min |
| aliens-dqn-reward2018-impala-b18 | 3 | 48.1 (79%, 74–84) | 34.5 (64%, 56–72) | 1.01M | 8 min |
| aliens-dqn-reward2018-long | 3 | 48.2 (91%, 86–94) | 13.9 (23%, 0–42) | 4.92M | 12 min |
| aliens-dqn-reward2018-p1-b18 | 3 | 40.9 (90%, 82–96) | 34.9 (69%, 62–78) | 1.01M | 2 min |
| aliens-dqn-reward2018-raw-mse-b18 | 3 | 44.4 (87%, 78–92) | 35.3 (79%, 60–96) | 1.01M | 2 min |
| aliens-dqn-rr0.5-b18 | 3 | 47.2 (35%, 10–72) | 35.5 (57%, 42–66) | 1.01M | 4 min |
| aliens-dqn-rr1.0-b18 | 3 | 53.0 (70%, 54–84) | 33.0 (46%, 30–54) | 1.01M | 7 min |
| aliens-ppo-b18 | 3 | 37.6 (82%, 78–86) | 35.0 (72%, 64–80) | 1.01M | 2 min |
| aliens-ppo-long | 3 | 45.9 (83%, 76–92) | 37.6 (82%, 80–84) | 4.99M | 9 min |
| aliens-ppo-outcome-only-b18 | 3 | 39.4 (78%, 74–82) | 39.3 (89%, 82–98) | 1.01M | 2 min |
| aliens-ppo-outcome-only-long | 3 | 43.5 (94%, 88–100) | 37.9 (84%, 76–90) | 4.98M | 11 min |
| aliens-ppo-reward2018-b18 | 3 | 37.6 (82%, 78–86) | 37.0 (82%, 82–82) | 1.01M | 2 min |
| aliens-rainbow-alllevels-long | 3 | 56.0 (81%, 70–88) | 43.8 (77%, 76–80) | 4.17M | 20 min |
| aliens-rainbow-b18 | 3 | 43.2 (66%, 62–72) | 39.1 (89%, 82–100) | 1.01M | 3 min |
| aliens-rainbow-impala-b18 | 3 | 54.7 (85%, 82–90) | 40.7 (78%, 70–94) | 1.01M | 11 min |
| aliens-rainbow-impala-long | 3 | 58.7 (93%, 90–96) | 41.4 (81%, 80–82) | 3.58M | 46 min |
| aliens-rainbow-long | 3 | 57.0 (93%, 90–98) | 37.7 (70%, 64–82) | 5.00M | 14 min |
| aliens-rainbow-reward2018-b18 | 3 | 39.6 (88%, 82–94) | 25.3 (31%, 0–64) | 1.01M | 5 min |
| aliens-rainbow-reward2018-impala-b18 | 3 | 43.4 (95%, 94–96) | 33.6 (69%, 52–78) | 1.01M | 12 min |
| aliens-rainbow-reward2018-long | 3 | 43.5 (91%, 88–94) | 30.0 (49%, 44–56) | 3.67M | 19 min |
| aliens-rainbow-rr0.5-b18 | 3 | 46.8 (63%, 50–70) | 37.9 (73%, 50–86) | 1.01M | 9 min |
| aliens-rainbow-rr1.0-b18 | 3 | 51.0 (65%, 48–78) | 34.0 (65%, 58–74) | 1.01M | 16 min |
| chase-best | 3 | 0.8 (0%, 0–0) | 0.0 (0%, 0–0) | 5.67M | 13 min |
| chase-ddqn2018 | 3 | 0.2 (0%, 0–0) | – | 3.32M | 63 min |
| chase-dqn-b18 | 3 | 0.0 (0%, 0–0) | 0.0 (0%, 0–0) | 1.77M | 5 min |
| chase-dqn-long | 3 | 0.6 (0%, 0–0) | 0.1 (0%, 0–0) | 4.75M | 12 min |
| chase-dqn-reward2018-long | 3 | 0.0 (0%, 0–0) | 0.0 (0%, 0–0) | 3.50M | 8 min |
| chase-ppo-b18 | 3 | 0.0 (0%, 0–0) | 0.0 (0%, 0–0) | 1.77M | 3 min |
| chase-ppo-reward2018-b18 | 3 | 0.0 (0%, 0–0) | 0.0 (0%, 0–0) | 1.77M | 3 min |
| chase-rainbow-20m | 1 | 1.7 (0%) | 0.4 (0%) | 20.00M | 36 min |
| chase-rainbow-b18 | 3 | 0.0 (0%, 0–0) | 0.0 (0%, 0–0) | 1.77M | 4 min |
| chase-rainbow-long | 3 | 0.0 (0%, 0–0) | 0.0 (0%, 0–0) | 3.67M | 8 min |
| chase-rainbow-reward2018-b18 | 3 | 0.0 (0%, 0–0) | 0.0 (0%, 0–0) | 1.77M | 8 min |
| chase-rainbow-reward2018-long | 3 | 0.0 (0%, 0–0) | 0.0 (0%, 0–0) | 3.25M | 14 min |
| deceptizelda-best | 3 | 2.0 (100%, 100–100) | 2.0 (100%, 100–100) | 3.25M | 18 min |
| deceptizelda-ddqn2018 | 3 | 1.6 (79%, 38–100) | – | 0.69M | 7 min |
| deceptizelda-dqn-b18 | 3 | 1.3 (67%, 0–100) | 1.3 (67%, 0–100) | 0.85M | 4 min |
| deceptizelda-dqn-long | 3 | 2.0 (100%, 100–100) | 2.0 (100%, 100–100) | 3.25M | 10 min |
| deceptizelda-dqn-reward2018-long | 3 | 1.9 (92%, 76–100) | 1.9 (88%, 64–100) | 3.25M | 10 min |
| deceptizelda-ppo-b18 | 3 | 2.0 (100%, 100–100) | 2.0 (100%, 100–100) | 0.81M | 3 min |
| deceptizelda-ppo-reward2018-b18 | 3 | 2.0 (100%, 100–100) | 2.0 (100%, 100–100) | 0.81M | 3 min |
| deceptizelda-rainbow-b18 | 3 | 1.0 (42%, 0–100) | 1.0 (46%, 0–100) | 0.85M | 6 min |
| deceptizelda-rainbow-long | 3 | 2.0 (100%, 100–100) | 2.0 (100%, 100–100) | 3.33M | 20 min |
| deceptizelda-rainbow-reward2018-b18 | 3 | 2.0 (100%, 100–100) | 2.0 (100%, 100–100) | 0.85M | 6 min |
| deceptizelda-rainbow-reward2018-long | 3 | 2.0 (100%, 100–100) | 2.0 (100%, 100–100) | 3.25M | 24 min |
| waferthinmints2018-best | 3 | 10.2 (31%, 24–37) | 10.2 (32%, 21–43) | 4.33M | 8 min |
| waferthinmints2018-ddqn2018 | 3 | 10.2 (35%, 32–38) | – | 1.61M | 20 min |
| waferthinmints2018-dqn-b18 | 3 | 7.7 (52%, 34–70) | 7.8 (51%, 38–68) | 1.60M | 2 min |
| waferthinmints2018-dqn-long | 3 | 9.5 (41%, 38–44) | 9.7 (37%, 34–42) | 4.25M | 9 min |
| waferthinmints2018-dqn-loss-only-b18 | 3 | 8.7 (53%, 44–58) | 8.4 (49%, 44–56) | 1.60M | 3 min |
| waferthinmints2018-dqn-outcome-b18 | 3 | 9.1 (47%, 38–52) | 9.3 (46%, 44–50) | 1.60M | 3 min |
| waferthinmints2018-dqn-outcome-g997-b18 | 3 | 5.7 (73%, 56–82) | 5.4 (79%, 68–96) | 1.60M | 3 min |
| waferthinmints2018-dqn-outcome-long | 3 | 9.5 (42%, 34–50) | 9.8 (40%, 28–52) | 4.67M | 9 min |
| waferthinmints2018-dqn-outcome10-b18 | 3 | 8.5 (59%, 46–72) | 8.1 (62%, 48–72) | 1.60M | 3 min |
| waferthinmints2018-dqn-outcome5-b18 | 3 | 9.2 (43%, 32–60) | 8.9 (44%, 32–66) | 1.60M | 3 min |
| waferthinmints2018-dqn-penalty-decay-b18 | 3 | 7.4 (51%, 34–68) | 7.6 (56%, 48–62) | 1.60M | 3 min |
| waferthinmints2018-dqn-reward2018-b18 | 3 | 7.5 (59%, 32–86) | 7.4 (55%, 20–92) | 1.60M | 3 min |
| waferthinmints2018-dqn-reward2018-long | 3 | 5.9 (79%, 68–94) | 6.4 (72%, 66–82) | 3.25M | 7 min |
| waferthinmints2018-ppo-b18 | 3 | 6.3 (73%, 72–74) | 5.8 (77%, 66–86) | 1.60M | 3 min |
| waferthinmints2018-ppo-reward2018-b18 | 3 | 2.3 (97%, 92–100) | 2.3 (97%, 92–100) | 1.60M | 3 min |
| waferthinmints2018-rainbow-b18 | 3 | 7.4 (57%, 42–86) | 6.8 (59%, 38–86) | 1.60M | 5 min |
| waferthinmints2018-rainbow-long | 3 | 9.5 (35%, 20–60) | 9.8 (39%, 28–54) | 3.50M | 10 min |
| waferthinmints2018-rainbow-reward2018-b18 | 3 | 2.7 (91%, 76–100) | 2.7 (93%, 82–100) | 1.60M | 5 min |
| waferthinmints2018-rainbow-reward2018-long | 3 | 5.8 (71%, 46–92) | 5.6 (68%, 50–82) | 5.00M | 13 min |

Long runs stopped early (no longer improving after 3M ticks): aliens-best-s0 at 9.0M, aliens-best-s1 at 7.3M, aliens-best-s2 at 6.8M, aliens-dqn-impala-long-s0 at 3.8M, aliens-dqn-impala-long-s1 at 4.5M, aliens-dqn-impala-long-s2 at 5.0M, aliens-dqn-reward2018-long-s0 at 4.8M, aliens-rainbow-alllevels-long-s1 at 4.3M, aliens-rainbow-alllevels-long-s2 at 3.3M, aliens-rainbow-impala-long-s0 at 4.0M, aliens-rainbow-impala-long-s1 at 3.3M, aliens-rainbow-impala-long-s2 at 3.5M, aliens-rainbow-reward2018-long-s0 at 4.5M, aliens-rainbow-reward2018-long-s1 at 3.2M, aliens-rainbow-reward2018-long-s2 at 3.3M, chase-best-s0 at 3.3M, chase-best-s1 at 3.8M, chase-dqn-long-s0 at 4.3M, chase-dqn-long-s2 at 5.0M, chase-dqn-reward2018-long-s0 at 3.8M, chase-dqn-reward2018-long-s1 at 3.5M, chase-dqn-reward2018-long-s2 at 3.3M, chase-rainbow-long-s0 at 3.3M, chase-rainbow-long-s1 at 3.5M, chase-rainbow-long-s2 at 4.3M, chase-rainbow-reward2018-long-s0 at 3.3M, chase-rainbow-reward2018-long-s1 at 3.3M, chase-rainbow-reward2018-long-s2 at 3.3M, deceptizelda-best-s0 at 3.3M, deceptizelda-best-s1 at 3.3M, deceptizelda-best-s2 at 3.3M, deceptizelda-dqn-long-s0 at 3.3M, deceptizelda-dqn-long-s1 at 3.3M, deceptizelda-dqn-long-s2 at 3.3M, deceptizelda-dqn-reward2018-long-s0 at 3.3M, deceptizelda-dqn-reward2018-long-s1 at 3.3M, deceptizelda-dqn-reward2018-long-s2 at 3.3M, deceptizelda-rainbow-long-s0 at 3.5M, deceptizelda-rainbow-long-s1 at 3.3M, deceptizelda-rainbow-long-s2 at 3.3M, deceptizelda-rainbow-reward2018-long-s0 at 3.3M, deceptizelda-rainbow-reward2018-long-s1 at 3.3M, deceptizelda-rainbow-reward2018-long-s2 at 3.3M, waferthinmints2018-best-s0 at 3.3M, waferthinmints2018-best-s1 at 5.0M, waferthinmints2018-best-s2 at 4.8M, waferthinmints2018-dqn-long-s0 at 4.5M, waferthinmints2018-dqn-long-s1 at 5.0M, waferthinmints2018-dqn-long-s2 at 3.3M, waferthinmints2018-dqn-outcome-long-s0 at 4.0M, waferthinmints2018-dqn-reward2018-long-s0 at 3.3M, waferthinmints2018-dqn-reward2018-long-s1 at 3.3M, waferthinmints2018-dqn-reward2018-long-s2 at 3.3M, waferthinmints2018-rainbow-long-s0 at 3.5M, waferthinmints2018-rainbow-long-s1 at 3.3M, waferthinmints2018-rainbow-long-s2 at 3.8M
