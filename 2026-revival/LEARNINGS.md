# Learnings, for the next revisit

Written in October 2026, for whoever opens this again in 2034 or later. What to check first, what broke last time, and what the project found that is worth knowing before starting.

## Coming back: checklist

1. Read [RESULTS.md](RESULTS.md) (findings) and this file. Nothing else is needed to start.
2. Install with the exact versions below. If they no longer install, see "What will have rotted".
3. `uv run python tests/golden.py check`: every recorded game must replay identically. This is the guarantee that the engine is still the 2018 games. A failure names the game, the first differing tick and whether the *state* (positions, score, game over) or only the *pixels* differ. State differences mean the games changed: stop there. Pixel-only differences are most likely a newer JDK drawing sprites slightly differently; the learning agents see pixels, so re-record (`tests/golden.py record`) only after checking a few frames by eye.
4. `uv run python tests/replay_check.py` and `tests/repeat_check.py`: replay sampling and action repeat are exact.
5. A short run: `uv run python -m train.modern --game aliens --preset dqn --ticks 100000 --eval-games 8 --name check` (about a minute in 2026; `results/check.json` appears at the end).
6. A real number to compare against: `uv run python -m tools.batch experiments/verify.txt` reruns every Aliens setup the blog shows; RESULTS.md, "The overnight check", has what it gave on 25 Oct 2026 and how long each took. RESULTS.md, "Speed", lists the speed benchmarks to rerun.
7. Tag the state you start from; tags here: `2018-*` (the 2018 code), `imp-NN` (each speed step), `batch-N` (result batches), `revival-2026` (the end of this round).

## Environment, October 2026

| Thing | Version |
|---|---|
| OS | Windows 11 Pro, build 26200.9711 |
| CPU / GPU / RAM | AMD Ryzen 7 7800X3D (8 cores) / NVIDIA RTX 4090 24 GB, driver 610.88 / 63 GB |
| Python | 3.14.8 via uv 0.13.1 (3.15 was out, but `torch.compile` didn't support it yet) |
| PyTorch | 2.14.1+cu130 (CUDA 13.0, cuDNN 9.24), wheel `torch-2.14.1+cu130-cp314-cp314-win_amd64.whl` from `https://download.pytorch.org/whl/cu130` |
| Triton (for `torch.compile`) | `triton-windows` 3.8.0.post29 |
| Others | numpy 2.5.4, gymnasium 1.4.0, Pillow 13.0.0; ruff 0.17.1, ty 0.0.89 |
| Java | Temurin OpenJDK 25.0.4.1 LTS, unpacked to `D:\Java\jdk-25.0.4.1+1` (`GVGAI_JDK` overrides) |
| Engine | GVGAI_GYM commit `15e0a86`, vendored in `engine/` with this project's changes; built by `engine/build.py` into `engine/build/classes-<hash of the sources>` |

All timings in RESULTS.md and IMPROVEMENTS.md are on this machine; they mean nothing on another one without re-measuring.

## What will have rotted

- **PyTorch's CUDA wheels.** Old CUDA builds disappear from download.pytorch.org over the years. `uv.lock` records the exact wheel URLs and hashes, and `pyproject.toml` the index (`pytorch-cu130`); if they are gone, point the index at a current CUDA build, `uv lock --upgrade`, and take the nearest PyTorch for a current Python and re-run the checklist. Nothing here depends on 2.14 specifically except `torch.compile` (optional: set `compile=False` in `ModernConfig`) and CUDA graphs (on since PyTorch 1.10).
- **Python.** Raise `requires-python` and `target-version` in `pyproject.toml`, `uv sync`, run ruff and the checklist.
- **Java.** The engine is plain Java with no dependencies, built by one `javac` call; point `GVGAI_JDK` at any recent JDK (only JDK 25 was tested).
- **Windows-only bits.** `tools/batch.py` uses `taskkill` and Windows idle priority; `bench/heap_growth.py` asks PowerShell for memory; the JDK default path is a Windows path. The rest is portable.
- **2018 history.** The 2018 code needs Keras 2 / TensorFlow 1 on Python 3.6 and will not install; it is kept for reading. `agents/ddqn2018.py` is the runnable version. The 2018 report lives in a separate repository (Dissertation-Report); its training logs, which the reproduction and the per-game budgets rest on, are copied into `results/2018/`.

## What the project found (the short version)

1. **Check what a trivial policy scores first.** Holding one button wins 82% of Aliens games, every Deceptizelda game and every Wafer Thin Mints game. Several "good" results, in 2018 and in 2026, were that policy. `baselines/trivial.py` runs these in a few minutes; run it before trusting any win rate on a new game.
2. **Reward the outcome, given enough experience.** +1 win / −1 loss (optionally + 0.1 × score) beat every hand-shaped reward at 5M ticks and held up on unseen levels. 2018's −0.5 penalty on every move of a lost game was the right fix for 2018's tiny budget, not a general one.
3. **Most of 2018's 12 hours was waiting.** A 25 ms sleep per tick (for human viewers), PNG files written and re-read every tick, one sample at a time through the network. None of it was learning.
4. **How often the agent acts mattered more than the method.** A move every 4 ticks (Atari's default) capped Aliens near 60 points; every tick or every 2 reached 79–83 at ~100% wins, level with or above MCTS (77.8), which needs little search on these games (50 iterations per move already gives 77.9 / 100%).
5. **Exploration is the open problem.** Nothing learned Deceptizelda's big exit (10 points) or a single win on Chase.

## How it was made fast, and where the time goes now

Full log with tags: [IMPROVEMENTS.md](IMPROVEMENTS.md). One game went from 34 to ~2,600 ticks/s; one modern DQN run trains at ~9,000 ticks/s alone, ~5,000 each with three at once.

- **The engine**: removing the per-tick sleep, sending raw frames instead of PNG files, greyscaling and halving on the server, repeating actions server-side, skipping drawing on repeated ticks, buffered sockets.
- **The env layer**: one Python process drives all 8 game servers over their sockets (no worker processes); finished games reset in parallel threads.
- **The learner**: replay memory and sampling on the GPU, the whole update recorded once as a CUDA graph and replayed, fused Adam, bf16, channels-last, `torch.compile` inside the graph, acting on its own CUDA stream with a copy of the network.
- **Now:** the standard network is limited by the 8 CPU cores running games (~25,000 ticks/s in total); the IMPALA network by the GPU (one update every 16 ticks, ~2.2 ms each). Parallel runs on one GPU take turns (Windows has no MPS), so several seeds in one process on separate streams was measured at only 1.3× and not built.

## Pitfalls hit, so they aren't hit again

- **The 2018 bugs were real and are kept in the replay**: rewards credited one move late; prioritised replay without importance weights; training only at game end; the evaluation used the weights from the last target sync. Fixing them changed little (48.2 / 80% vs 50.1 / 81%).
- **2018's undocumented settings** (from the `2018-ddqn-per` code, not the report): Keras learning-rate decay 1e-4, MSE loss, ε held at 1.0 through 20k warm-up stacks then down to 0.1 over ~1000 episodes, 2 × stacks / 32 minibatches at each game end, target sync every 5000 stacks checked at game end, new transitions prioritised by their current error.
- **Wafer Thin Mints' rules changed after 2018**: GVGAI_GYM's later file added a −20 death penalty. `waferthinmints2018` is the June 2018 file, and its timeout-win rule was tab-indented, so the parser ignored it until re-indented.
- **Deceptizelda's level 1** is a different size; 2018's network could not have played it, so every agent uses levels 0 and 2.
- **A sprite-cache leak** filled the server's heap after a few thousand Deceptizelda games; the client then spun on the closed socket instead of failing. Now a weak-keyed cache, and a closed socket raises.
- **The engine disqualifies a move that takes over 500 s** (`ACTION_TIME_DISQ`): a slow first `torch.compile` in the middle of a game killed runs. Autotuned compile is off for that reason.
- **Runs starting together corrupted the shared `torch.compile` cache**; each process now gets its own.
- **Results record the commit at the start of a run**, not the end; a commit made while a run trains is not the code it ran.
- **The golden test** compares every tick's state and pixels. Re-recording from `tests/pristine_engine.py` in October 2026 gave identical ticks, but the games the test cuts off before they end were labelled `NO_WINNER` instead of `PLAYER_LOSES`. The recorded files keep the original labels, which the current engine matches. Results don't depend on it: evaluation and 2018's reward count every game without a win as a loss. Cause not investigated.
- **Early stopping** must not stop on a plateau before ~3M ticks: Rainbow's curve flattens and then climbs again.
- **An evaluation can't be replayed exactly**: the same seed gives the same games, but a compiled network or another cuDNN algorithm shifts the values slightly, near-tied moves flip, and about 80% of an Aliens evaluation's scores change (its win rate doesn't). Record what you need while evaluating; `agents/evaluate.py` now keeps how each game ended (`ends`) and its length (`ticks`).
- **The 2018 agent is tested on different games**: `train/ddqn2018.py` evaluates on its training environment after training, not on the modern agents' seed + 7 games, so its comparisons are unpaired (fine for Welch's t-test, not for game-by-game claims).
- **A fixed time limit killed healthy runs**: tools.batch used to kill any run past 2 hours, and three 3-hour runs (a move every tick + IMPALA) died at 70% with nothing saved. The guard is now on a run's log going quiet, paused runs are exempt, and both trainers checkpoint (`train.modern` every 20 minutes, `train.ddqn2018` every 15).

## What I'd do next

- A larger outcome (±10) with 0.1 × score, so a game's points (~+4) can never outweigh a death: the ~1% of games the best setups lose are all deaths taken while scoring (RESULTS.md, "Winning and scoring together").
- Why held-out levels lag with a light score weight (0.05: 86%, one seed 73%) while a move every tick holds 100%.
- An exploration bonus (Random Network Distillation) for Deceptizelda's big exit and Chase.
- An equal-compute comparison (the IMPALA network costs 2–3× the wall time per tick; this project compared at equal ticks).
- PPO with the IMPALA network; more GVGAI games, to see whether the trivial-policy problem is common.
