# Working on this repo

- `2018-dissertation/` and the `2018-*` tags are frozen; new work goes in `2026-revival/` (or a new dated folder for a later revival). Coming back after a long gap: [2026-revival/LEARNINGS.md](2026-revival/LEARNINGS.md) first.
- Use the newest tooling everywhere; the point of the project is to be modern and fast. Python: the newest release PyTorch's compiler (`torch.compile`) supports, via uv (3.14 as of 19 Oct 2026; Colin chose this over 3.15 so training can be compiled), ruff for lint and format, ty for types. Java: latest LTS JDK, engine built by `engine/build.py` (no Gradle; agreed 18 Oct 2026).
- Approved 18 Oct 2026: GPU runs go through `lease` without asking each time; commit locally to master as steps land, one commit (and an `imp-NN` tag for speed steps) per improvement; never push or publish (the repo is public).
- Fable (a separate reviewer model) reviews major decisions and anything written for others before it is acted on or shown.
- Speed first: exhaust realistic speed gains on game ticks and training before long batches; real overall training time is the goal, and heavier engineering is fine if learning quality stays the same. Try ideas with `tools/screen.py` before long runs.
- Every engine change must keep `tests/golden.py check` identical; every learner change is checked against a training curve from before it.
- RAM is shared with Cortex (~39 GB for its Strata model): stop every game server when a test ends; reserve big RAM through `lease`. Cortex may be stopped entirely while training runs (`lease free --force --name cortex --stop-job`); restart it afterwards from its Startup shortcut.
- Results: one JSON per run in `2026-revival/results/`, its run list in `2026-revival/experiments/`; RESULTS.md and IMPROVEMENTS.md are kept current as results land.
