"""Run a list of training commands, a few at a time; skip any whose result file already exists.

    uv run python batch.py experiments/phase2.txt --parallel 6

Each line of the run list is the arguments for a training script (train2018.py, train_modern.py,
train_ppo.py), starting with the script name, e.g.
`train2018.py --game chase --seed 1 --name chase-ddqn2018-s1 --device cuda`.
Wrap in `lease run` for the GPU; a stopped batch picks up where it left off (train2018 resumes from
its checkpoint).
"""

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
IDLE = 0x40  # Windows IDLE_PRIORITY_CLASS; the run's game servers inherit it


def name_of(args: list[str]) -> str:
    return args[args.index("--name") + 1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("jobs")
    p.add_argument("--parallel", type=int, default=4)
    p.add_argument("--max-minutes", type=float, default=120, help="kill a run that takes longer (hang guard)")
    p.add_argument(
        "--low-priority", action="store_true", help="idle priority (and its game servers): for while Colin games"
    )
    a = p.parse_args()
    jobs = [shlex.split(ln) for ln in Path(a.jobs).read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    todo = [j for j in jobs if not (HERE / "results" / f"{name_of(j)}.json").exists()]
    print(f"{len(todo)} of {len(jobs)} jobs to run", flush=True)
    beat = HERE / "runs" / f"batch-{os.getpid()}.alive"  # see gvgai/heartbeat.py
    beat.touch()
    env = {**os.environ, "GVGAI_HEARTBEAT": str(beat)}
    running: dict[str, subprocess.Popen] = {}
    started: dict[str, float] = {}
    failed = []
    while todo or running:
        while todo and len(running) < a.parallel:
            j = todo.pop(0)
            n = name_of(j)
            log = open(HERE / "runs" / f"{n}.log", "a")  # noqa: SIM115 - closed with the process
            running[n] = subprocess.Popen([sys.executable, "-u", *j], cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, env=env,
                                          creationflags=IDLE if a.low_priority else 0)  # fmt: skip
            started[n] = time.time()
            print(f"start {n}", flush=True)
        time.sleep(5)
        beat.touch()
        for n, pr in list(running.items()):
            if pr.poll() is None and time.time() - started[n] > a.max_minutes * 60:
                print(f"TIMEOUT {n} after {a.max_minutes:.0f} min: killing it and its game servers", flush=True)
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(pr.pid)], capture_output=True)
                pr.wait()
            if pr.poll() is not None:
                ok = pr.returncode == 0 and (HERE / "results" / f"{n}.json").exists()
                print(f"{'done' if ok else 'FAILED'} {n} (exit {pr.returncode})", flush=True)
                if not ok:
                    failed.append(n)
                del running[n]
    print(f"finished; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
