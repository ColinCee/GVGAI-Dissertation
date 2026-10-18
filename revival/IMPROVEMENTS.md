# Improvements, in order

One row per change that landed. Speeds are game ticks per second for one Aliens game with random play, unless the row says otherwise.

| # | Area | Change | Before → after | How measured | Commit |
|---|---|---|---|---|---|
| 1 | Tooling | Python 3.6 + Keras 2 / TensorFlow 1 → Python 3.15 + PyTorch 2.14 (CUDA 13); JDK 8 → JDK 25; ruff and ty | 2018 stack no longer installs → everything installs with `uv sync` | — | `3000316` |
| 2 | Engine speed | Removed the engine's 25 ms sleep per tick (it paces the game for people watching; 2018 had it too) | 34 → 75 ticks/s | `smoke.py`, 5–20 games | `632bc10` |
| 3 | Engine speed | Python decodes each frame in memory instead of writing it to disk and reading it back | 75 → 136 ticks/s | cProfile + `smoke.py` | `632bc10` |
| 4 | Engine speed | Java sends the PNG it already encoded instead of saving, re-reading and re-encoding it | 136 → ~200 ticks/s | `smoke.py`, 40 games | `632bc10` |
| 5 | Fidelity | Rebuilt the 2018 Wafer Thin Mints rules (today's version adds a −20 death penalty) from GVGAI_GYM's June 2018 file, with a scoring typo fixed and its win-at-1500-ticks rule re-indented (tabs made the engine skip it silently) | random deaths score −10 → 11 and survivors win at tick 1500, both as in 2018's data | `smoke.py` vs 2018 `reward_history.csv` | `632bc10` |
| 6 | Fidelity | Golden test: four games replayed with fixed seeds; every later change must give the same state and pixels on every tick as the unmodified engine | — | `tests/golden.py` | `7d60156` |
| 7 | Engine speed | Sprites scaled once and cached instead of on every draw | 214 → 233 ticks/s | `bench.py` (30 s, after warm-up; used from here on) | `a24a0f3` |
| 8 | Engine speed | Server's output buffered and sent in one write, Nagle off | 233 → 274 ticks/s | `bench.py` | `a24a0f3` |
| 9 | Engine speed | Frames sent as raw RGB inside the message: no PNG encode, no frame file written each tick (reusing one JSON encoder alone changed nothing) | 274 → 1,017 ticks/s | `bench.py` | `a24a0f3` |
| 10 | Engine speed | Serial GC and a 128 MB heap per game server | 686 → 122 MB RAM per server, same speed; 16 servers: 854 → 6,660 ticks/s total (Phase 0 → now) | `runs/mem.py`, `bench_parallel.py` | `1e8a08c` |
| 11 | Fidelity | 2018 final agent ported to PyTorch with every setting and quirk kept (raw 0–255 input, reward credited one action late, training only at game end); a bug in the port (4× too many updates) caught by review | Aliens win % per 500 games: 2018 9/34/61/72 → 2026 8/29/61/75 | `train2018.py`, seed 0, vs 2018's `reward.xlsx` | `e98135f` |
| 12 | Training speed | Same 2018 agent, same 1M game ticks on Aliens, today's PC (CPU only) | ~12 h → 34 min (21× faster) | `results/aliens-ddqn2018-s0.json` | `e98135f` |
