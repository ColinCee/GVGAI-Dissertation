"""Run a list of training commands, a few at a time; skip any whose result file already exists.

    uv run python -m tools.batch experiments/phase2.txt --parallel 6

Each line of the run list is the arguments to `python`, run from this folder, e.g.
`-m train.ddqn2018 --game chase --seed 1 --name chase-ddqn2018-s1 --device cuda`.
Wrap in `lease run` for the GPU; a stopped batch picks up where it left off (both trainers resume
from their checkpoints). A run whose log goes quiet for --hang-minutes is killed as hung, unless
tools.pause has paused it, and started once more from its checkpoint.
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from gvgai.games import ROOT

HERE = ROOT
IDLE = 0x40  # Windows IDLE_PRIORITY_CLASS; the run's game servers inherit it


def name_of(args: list[str]) -> str:
    return args[args.index("--name") + 1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--hang-minutes", type=float, default=30, help="kill a run whose log is quiet this long")
    p.add_argument(
        "--low-priority", action="store_true", help="idle priority (and its game servers): for while Colin games"
    )
    a = p.parse_args()
    jobs = [shlex.split(ln) for ln in Path(a.jobs).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    todo = [j for j in jobs if not (HERE / "results" / f"{name_of(j)}.json").exists()]
    print(f"{len(todo)} of {len(jobs)} jobs to run", flush=True)
    (HERE / "runs").mkdir(exist_ok=True)
    beat = HERE / "runs" / f"batch-{os.getpid()}.alive"  # see gvgai/heartbeat.py
    beat.touch()
    env = {**os.environ, "GVGAI_HEARTBEAT": str(beat)}
    running: dict[str, subprocess.Popen] = {}
    args: dict[str, list[str]] = {}
    quiet_since: dict[str, float] = {}
    started: dict[str, float] = {}
    retried: set[str] = set()
    last_paused = 0.0  # when tools/pause.py last had runs paused
    failed = []
    while todo or running:
        while todo and len(running) < a.parallel:
            j = todo.pop(0)
            n = name_of(j)
            log = open(HERE / "runs" / f"{n}.log", "a")  # noqa: SIM115 - closed with the process
            running[n] = subprocess.Popen([sys.executable, "-u", *j], cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, env=env,
                                          creationflags=IDLE if a.low_priority else 0)  # fmt: skip
            args[n] = j
            started[n] = time.time()
            print(f"start {n}", flush=True)
        time.sleep(5)
        beat.touch()
        paused = (HERE / "runs" / "paused").exists()  # see tools/pause.py
        if paused:
            last_paused = time.time()
        for n, pr in list(running.items()):
            log_age = time.time() - (HERE / "runs" / f"{n}.log").stat().st_mtime
            if paused:
                quiet_since.pop(n, None)  # a paused run is quiet on purpose
            elif log_age > 60:
                # its quiet time counts from its last output, the end of the pause or its own start, whichever
                # is latest (a rerun appends to the old run's log, whose last write can be hours old)
                quiet_since.setdefault(n, max(time.time() - log_age, last_paused, started[n]))
            else:
                quiet_since.pop(n, None)
            if pr.poll() is None and n in quiet_since and time.time() - quiet_since[n] > a.hang_minutes * 60:
                print(f"HUNG {n}: no output for {a.hang_minutes:.0f} min, killing it and its game servers", flush=True)
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(pr.pid)], capture_output=True)
                pr.wait()
            if pr.poll() is not None:
                ok = pr.returncode == 0 and (HERE / "results" / f"{n}.json").exists()
                print(f"{'done' if ok else 'FAILED'} {n} (exit {pr.returncode})", flush=True)
                del running[n]
                quiet_since.pop(n, None)
                if not ok and n not in retried and (HERE / "runs" / n / "checkpoint").exists():
                    retried.add(n)
                    todo.insert(0, args[n])
                    print(f"retrying {n} from its checkpoint", flush=True)
                elif not ok:
                    failed.append(n)
    print(f"finished; failed: {failed or 'none'}")
    beat.unlink(missing_ok=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
