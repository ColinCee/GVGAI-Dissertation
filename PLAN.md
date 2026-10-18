# Revival plan: GVGAI DQN dissertation, 2018 → 2026

Hobby goal: rerun the 2018 dissertation (DQN → DDQN → PER → Dueling on GVGAI games, vs MCTS and human) on today's software and hardware, check the results reproduce, then see how much better modern methods do.

## What exists

| Thing | Where | State |
|---|---|---|
| Final 2018 agent (DDQN + PER + loss-penalty reward) | branch `origin/ddqn_per` (Apr 2018) | Reference implementation. Keras 2 / TF 1, Python 3.6. |
| Other variants | `origin/dqn`, `origin/ddqn`, `origin/dueling_network` | One branch per variant. |
| `master` | `src/` | Older plain-DQN version; not the final code. |
| Eval agent | `ddqn_per:src/QLearner-Test/Agent.py` | Loads weights, ε = 0, levels picked at random from 0–2. |
| Report | `D:\Claude Projects\Dissertation-Report\report\Report.tex` | Results tables (below). |
| Raw per-episode data | `Dissertation-Report\spreadsheet\*.csv`, `report\images\**\*.xlsx` | 4 columns, probably clipped episode reward, steps, win, ε (older stats format than `ddqn_per`'s 6-column one; confirm per file). Gives target learning curves. Not game score, so compare curves on clipped reward. |
| Game engine | GVGAI Java (EssexUniversityMCTS/gvgai, last push 2018) | Code here is only the Python client; it ran pasted inside that repo, talking over a socket to a Java server, frames as PNG. |

2018 numbers to reproduce:

- Aliens, plain DQN, win % by episode range: 1–500 6%, 500–1000 12%, 1000–1500 29%, 1500–2000 35%. 2000 episodes took ~12 h.
- Aliens, DDQN + PER + new reward: 4%, 22.2%, 58.0%, 68.2%.
- Final table (50 eval episodes, avg score (win %)):

| | Aliens | Chase | Deceptizelda | Wafer Thin Mints |
|---|---|---|---|---|
| Human (10 eps) | 42.6 (100%) | 1.2 (0%) | 10.6 (80%) | 11 (0%) |
| DDQN+PER+reward | 56.3 (82.6%) | 0.7 (0%) | 1.99 (99%) | 10.2 (33%) |
| MCTS | 79.2 (100%) | 3.5 (3.7%) | 1.91 (84.4%) | 9.9 (75.5%) |

Key 2018 settings, as in the `ddqn_per` code (the report leaves some out):

- γ 0.99, batch 32, replay 100k, warmup 20k stacks, 4 frames per stack with action repeat, 2× downscale to greyscale.
- Adam lr 2.5e-4, clipnorm 1, **Keras `decay=1e-4`** (lr shrinks to ~⅓ by episode 2000), **MSE loss** (not Huber).
- ε held at 1.0 through warmup, then 1 → 0.1 linearly over 1000 post-warmup episodes (reaches 0.1 around episode 1100–1250).
- Training only at episode end (2 × stacks / 32 minibatches); target sync every 5000 stacks, checked only at episode end. Dueling streams 256 wide.
- PER: α 0.6, ε 1e-4, **no importance-sampling weights**, new transitions get their current TD error as priority (not max).
- Reward clipped to ±1, −0.5 on every transition of a lost game; hitting the 2000-tick limit counts as a loss.
- **Bug: rewards are shifted one decision step** (`ddqn_per:src/QLearner/Agent.py` ~85–92: the stored reward is the score change from the *previous* action; at game end the last interval's change is overwritten by the ±1 outcome). This changes results.
- Eval used the weights saved at the **last target sync**, not the end of training.
- The 12 h figure was plain DQN on a GTX 980 (4 GB).

Machine now: RTX 4090 24 GB, Ryzen 7 7800X3D, 63 GB RAM. In 2018 the replay memory had to be kept small "due to hardware constraints".

## Where to work

- Same repo, on `master`, new work in `revival/`. Old `src/` untouched. 2018 code frozen under local tags `2018-dqn`, `2018-ddqn`, `2018-ddqn-per`, `2018-dueling`.
- The game engine (GVGAI_GYM's Java, Apache 2.0) is copied into `revival/engine/` at commit `15e0a86`, so speed fixes are ordinary commits next to the code that uses them.
- Commit locally as each step lands. Nothing is pushed: the repo is public, so pushing is his call at the end.

## How the work runs

- Runs without his input from approval to final results. He is only asked again for anything public, money, or a real blocker.
- Fable (a separate reviewer model) reviews each decision point marked **[Fable]** below before it is acted on; its advice and what was done go in the improvement log.
- RAM is tight: Cortex's local model needs ~39 GB of the 63, so runs stay within ~16 GB (replay stores each frame once) and stop every game server when they end.
- GPU and big RAM through `lease run`, so he can stop a run any time with `lease free --force --stop-job`. Every training run checkpoints every 15 minutes and resumes after being stopped. Every run has a timeout just above its expected time.
- Long runs go overnight or while he is away, judged from `lease show`; a smoke run of a few minutes comes before every long run.
- Venvs, checkpoints, and run data on D:, outside git. Small result files (JSON/CSV summaries) are committed.

## Tooling

- Python 3.15 via uv (released 9 Oct 2026); PyTorch 2.14 from PyTorch's CUDA 13 index (3.15 wheels exist there, not yet on PyPI). If any needed package has no 3.15 build, drop to 3.14 and log it, without asking.
- ruff (lint + format) must pass on every commit; ty (types, still pre-1.0) is run and reported but doesn't block.
- Java: JDK 25 LTS (Temurin), engine built by one short script (no Gradle: a plain 443-file tree with no dependencies doesn't need it, and build tooling doesn't make games faster).

## Phase 0: one game running — done

Results and setup in [revival/README.md](revival/README.md): all four games run, frames and actions match 2018, ~200 ticks/s per game (from 34).

## Phase 1: fast environment

Goal: the game is no longer what holds training back. Stop when the next idea gains under 15%, when the learner rather than the games is the bottleneck, or after 3 working days, whichever is first.

1. **Golden test first.** The server picks a fresh random seed per game (`LearningMachine.java`, `new Random().nextInt()`), so first add a way to set it (the only engine change allowed before the test exists). Then record games on the engine with fixed seeds and scripted actions: game state, score and win per tick, plus decoded pixels (not PNG bytes, which can differ between JDKs). Every speed change must reproduce them exactly, or it is reverted. This is what keeps the 2018 games the same games.
2. Then, measuring each one alone: raw pixels instead of PNG; skip the frame file; binary messages instead of JSON; JVM flags; many games inside one JVM; a vectorised env API for Python.
3. **[Fable]** on the order and on anything that touches game code rather than plumbing.

## Phase 2: faithful 2018 replay

As described above: port `ddqn_per` to PyTorch with every 2018 setting and bug kept.

These runs are small (1.5M-parameter net, env-bound): they run on CPU, several at once, no GPU lease.

1. Gate run: final agent on Aliens, 2000 episodes, one seed. Pass if the episode 1500–2000 win rate is within ±10 points of 2018's 68.2%, and 1000–1500 within ±10 of 58.0%. **[Fable]** on the verdict; if it fails, find why before going on.
2. Final agent × 4 games × 3 seeds.
3. The 2018 ablation (plain DQN, DDQN, Dueling on Aliens, 3 seeds) only if the gate passed within ±5 points; otherwise skipped and logged.
3. Timing: plain DQN on Aliens with 2018's design vs the 12 h of 2018, reported per 1000 ticks and per 1000 updates as well as total.

## Phase 3: modern agents

Each is one step in the improvement log, measured on its own against the step before:

1. Fix the 2018 bugs (reward timing, PER weights), nothing else.
2. Modern DQN defaults: update every step, Huber, no lr decay, 1M replay.
3. Rainbow (CleanRL's `rainbow_atari.py` as the reference).
4. PPO with many games in parallel.

Modern agents act every tick (no 4-tick repeat) unless the method's own defaults say otherwise; whichever is used is logged.
5. Stretch, if time allows: newer sample-efficient methods (e.g. BBF-style scaling), and generalisation (train levels 0–2, test 3–4).

**[Fable]** before each method is chosen and after each finishes.

## How the comparison works

Same for every row so the numbers can sit in one table:

- Eval: 50 games per seed, ε = 0 (and ε = 0.01 as a second column for modern agents), levels 0–2, plus levels 3–4 held out.
- 3 seeds per row; reported as mean and spread.
- Budgets are counted in game ticks, so an agent that acts every tick gets 4× the decisions of 2018's; that is part of what's being compared.
- **2018's budget**: the ticks 2018's final agent used in its first 2000 episodes on that game (Aliens: 1,014,134, from `DDQN + PER Reward.csv`).
- **Long budget**: up to 5M ticks, stopped early when the win rate gains under 2 points over the last 20% of training.
- Rows: Human (2018), MCTS (2018), MCTS 2026 at a fixed iteration count and at 40 ms per move, DDQN (2018 report), DDQN replay (2026), then each modern step.
- Plus: learning curves over the 2018 CSVs, and a wall-clock table (2018 vs replay vs modern).

## The improvement log and blog post

- `revival/IMPROVEMENTS.md`: one row per change that lands, written as it lands: number, area (tooling / engine speed / training speed / agent), what changed, before → after, how measured, commit.
- `revival/results/`: one summary file per run; one script makes every table and chart from them, so the post's numbers can be regenerated.
- At the end: a draft post in `revival/BLOG.md`, scannable: headline numbers, then the improvements in order, each with its before → after. Publishing is his call.

## Rough cost

- Phase 1: a few days of work, CPU only.
- Phase 2: CPU only, about 1.5 h per run at today's speed, several in parallel.
- Phase 3 target: the whole modern comparison in **under a day of real time**, and minutes, not hours, to try an idea. At today's speed it would be 70–140 h, so Phase 1's speed-up decides it. After Phase 1, measure, then set the long budget, seeds and methods to fit the day; log the choice. Try ideas on short runs (one game, one seed, a fraction of the budget) before full runs.

## Risks

- Exact reproduction may be off (dated engine version, undocumented details); the 2018 CSVs are the referee.
- Python 3.15 or ty may be too new for some package; fall back one version and log it.
- Many-games-per-JVM may hit shared static state in the engine (it uses static fields); the golden test catches it.
