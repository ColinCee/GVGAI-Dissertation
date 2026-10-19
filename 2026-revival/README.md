# 2026 revival

The 2018 dissertation rebuilt on 2026 software: the GVGAI engine made ~75× faster per game, the 2018 agent ported to PyTorch with every 2018 setting and bug kept, and today's methods compared on the same games and budgets. Findings: [RESULTS.md](RESULTS.md). Every speed step, with a tag to check it out: [IMPROVEMENTS.md](IMPROVEMENTS.md). Coming back later: [LEARNINGS.md](LEARNINGS.md).

## Setup (Windows)

1. [uv](https://docs.astral.sh/uv/), and JDK 25 (Temurin zip unpacked to `D:\Java\jdk-25.0.4.1+1`, or set `GVGAI_JDK`).
2. `uv sync` in this folder: Python 3.14 (the newest `torch.compile` supports), PyTorch from PyTorch's CUDA 13 index. Exact versions are in `uv.lock`.
3. `uv run python tests/golden.py check`: the engine compiles itself on first use ([engine/build.py](engine/build.py)) and must replay the recorded games identically.
4. Before committing: `uv run ruff format . && uv run ruff check .`; `uv run ty check` is advisory (most of its warnings are gaps in PyTorch's type hints).

## Layout

Run everything from this folder as modules, e.g. `uv run python -m train.modern --game aliens --preset dqn-outcome-score01 --ticks 5000000 --name my-run`.


| Path | What |
|---|---|
| `agents/` | The learners: `ddqn2018.py` (the 2018 agent, every setting and bug kept), `modern.py` (DQN, Rainbow and their pieces; replay and the whole update on the GPU), `evaluate.py` (the evaluation every agent is scored with) |
| `train/` | One entry point per family: `-m train.ddqn2018` (variants dqn / ddqn / dueling, `--fix-bugs`), `-m train.modern --preset ...`, `-m train.ppo` (CleanRL's Atari defaults) |
| `baselines/` | Players that don't learn: `-m baselines.mcts` (40 ms or a fixed number of iterations per move), `-m baselines.trivial` (one button held all game, random play) |
| `tools/` | `-m tools.screen` (several setups × seeds at once, one table), `-m tools.batch` (a run list a few at a time, skipping finished runs), `-m tools.report` (builds [results/summary.md](results/summary.md)), `-m tools.smoke` (random play: does the engine run, how fast), `-m tools.record` (one game as video, poster and per-move JSON with Q-values, for showing agents play) |
| `experiments/` | The run lists behind every reported number: `uv run python -m tools.batch experiments/<file>.txt` |
| `results/` | One JSON per run (evaluation, timing, config, commit); `results/curves/` the training logs behind reported curves; `results/2018/` the 2018 training data |
| `gvgai/` | The Python side of the engine: env, many games at once, shared game list |
| `engine/` | GVGAI_GYM's Java engine and games (commit `15e0a86`) with this project's speed changes |
| `tests/` | Golden replay of the engine, action repeat, replay sampling |
| `bench/` | The measurements cited in IMPROVEMENTS.md |
| `runs/` | Per-run training logs and checkpoints, created by each run (not in git; this round's 57 GB were deleted, keeping the reported curves in `results/curves/`) |
| `weights/` | Trained networks kept for showing agents play: the best modern DQN, the IMPALA DQN and the 2018 agent replay, all on Aliens (not in git, 45 MB) |

## Engine changes

[engine/](engine/) was imported unmodified at GVGAI_GYM commit `15e0a86`, then changed (each change is a row in IMPROVEMENTS.md, and the golden test proves the games still play the same):

- the 25 ms sleep in every tick is configurable (`-Dgvgai.longDelay`; the env passes 0);
- frames are sent raw and can be greyscaled and halved on the server (`gvgai.rawFrames`, `gvgai.grayHalf`), drawing is skipped while an action repeats, and actions can be repeated server-side (`ACTION#IMAGE#k`);
- a fixed seed (`gvgai.seed`), a tick cap (`gvgai.maxTicks`) and MCTS budgets (`gvgai.actionTime`, `gvgai.mctsIters`);
- Wafer Thin Mints levels 1–4 added from GAIGResearch/GVGAI, and `waferthinmints2018`: the June 2018 rules the dissertation ran (GVGAI_GYM commit `3dbe18b`), with the `changeScore` → `scoreChange` typo fixed and the timeout rule re-indented so it applies.

Python client changes: [gvgai/vendor/README.md](gvgai/vendor/README.md).

## Protocol

- Budgets are in game ticks: "2018's ticks" is what 2018 used per game ([results/2018/budgets.json](results/2018/budgets.json)); long runs go to 5M with early stopping.
- Evaluation: 50 games (100 for the best-of rows), greedy, levels 0–2, plus held-out levels 3–4. Deceptizelda uses levels 0 and 2 only: level 1 is a bigger map than 2018's network could take.
- A 50-game win rate is ±10 points; 2018's two runs of the same agent differ by 10–17 points per 500-game block.
- The 2000-tick limit counts as a loss in evaluation and in 2018's reward, and as a truncation (bootstrapped) for modern learners.
