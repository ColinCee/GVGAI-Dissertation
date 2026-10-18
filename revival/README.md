# Revival: the 2018 GVGAI DQN dissertation on 2026 software

Plan: [../PLAN.md](../PLAN.md). The 2018 code is frozen under tags `2018-dqn`, `2018-ddqn`, `2018-ddqn-per`, `2018-dueling`.

## Setup (Windows)

1. JDK 21 and [uv](https://docs.astral.sh/uv/).
2. Game engine, pinned and patched:
   ```bash
   git clone https://github.com/rubenrtorrado/GVGAI_GYM.git D:/Repos/GVGAI_GYM
   cd D:/Repos/GVGAI_GYM
   git checkout 15e0a86dcaa138721409b07ed970255485e150d6
   git apply "D:/Claude Projects/GVGAI-Dissertation/revival/gvgai/gvgai_gym.patch"
   python build.py
   ```
   Set `GVGAI_GYM_DIR` if it lives elsewhere.
3. `uv sync` in this folder.
4. `uv run python smoke.py --game aliens --episodes 5 --repeat 4`

The patch ([gvgai/gvgai_gym.patch](gvgai/gvgai_gym.patch)):
- makes the engine's per-tick sleep configurable (`-Dgvgai.longDelay`, default still 25 ms; our env passes 0);
- sends the frame PNG it already encoded instead of writing, re-reading and re-encoding it each tick;
- adds Wafer Thin Mints levels 1–4 (GVGAI_GYM ships only level 0), taken from [GAIGResearch/GVGAI](https://github.com/GAIGResearch/GVGAI), whose level 0 is identical;
- adds `waferthinmints2018`, the rules as the dissertation ran them (see below).

Python client changes are listed in [gvgai/vendor/README.md](gvgai/vendor/README.md). The env is [gvgai/env.py](gvgai/env.py): the game's own actions (no extra no-op), random level 0–2 per game, 2000-tick limit counted as a loss.

## Phase 0 results (18 Oct 2026)

| Game | Actions | Frame (2018 downscaled) | Notes |
|---|---|---|---|
| Aliens | USE, LEFT, RIGHT | 110×300 (55×150) | matches 2018 exactly |
| Chase | LEFT, RIGHT, DOWN, UP | 110×240 (55×120) | |
| Deceptizelda | USE + 4 moves | 120×130 (60×65) | |
| Wafer Thin Mints | 4 moves | 100×100 (50×50) | use `waferthinmints2018`, see below |

Random play on Aliens with 2018's 4-tick action repeat wins 3/40 (7.5%); 2018's random phase won 4–6%. Without the repeat it wins ~45%, so the repeat matters.

Speed, Aliens, random play, one game:

| Step | Ticks/s |
|---|---|
| As shipped | 34 |
| Engine sleep removed | 75 |
| + Python decodes frames in memory | 136 |
| + Java sends the PNG it already made | ~200 |

The 2018 engine had the same 25 ms sleep per tick, so roughly 7 of the original 12 hours for 2000 episodes were the game pausing on purpose.

Several games at once (separate processes): 1 → 196, 4 → 325, 8 → 408, 16 → 854 ticks/s in total. Scaling is poor. Phase 2 ideas: JVM flags through `JAVA_OPTS` (`-XX:ActiveProcessorCount=1 -XX:+UseSerialGC -XX:TieredStopAtLevel=1`), and dropping the per-tick frame file the engine still writes.

**Wafer Thin Mints rules.** In 2018's data (`Dissertation-Report/report/images/different games/new reward/wafer/reward_history.csv`: episode, game score, clipped reward, ticks, win, ε) every win is at exactly tick 1500 and deaths score +11. GVGAI_GYM's current file (Oct 2018) adds a −20 death penalty, so random deaths score −10. `waferthinmints2018` is its June 2018 file (commit `3dbe18b`) with one typo fixed (`changeScore` → `scoreChange`; without it nothing scores). Random deaths there score 11, matching 2018.
