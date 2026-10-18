# Working on this repo

- The 2018 dissertation code is frozen under the `2018-*` tags and old branches; new work goes in `revival/` (see [PLAN.md](PLAN.md)).
- Use the newest tooling everywhere; the point of the project is to be modern and fast. Python: latest release via uv, ruff for lint and format, ty for types. Java: latest LTS JDK, engine built by a short script (no Gradle; agreed 18 Oct 2026).
- Approved 18 Oct 2026: run [PLAN.md](PLAN.md) end to end without checking in. GPU runs go through `lease` without asking each time; commit locally to master as steps land; never push or publish. Come back only with results or a real blocker.
- Fable reviews each **[Fable]** decision point in the plan before it is acted on.
- Iteration speed matters as much as results: aim for the full modern comparison in under a day of real time, and short runs to try ideas.
- Colin (18 Oct): exhaust realistic speed gains on both game ticks and training before starting long training batches; measure on a quiet machine. Real overall training time is the goal, not effort: heavier engineering is fine if learning quality stays the same.
- RAM is shared with Cortex (~39 GB): stop every game server when a test ends; reserve big RAM through `lease`. Colin (18 Oct): Cortex (and its Strata model) may be stopped entirely while training runs; speed and more ticks come first. Restart it afterwards.
