"""Training runs started by tools/batch.py stop when the batch is stopped (e.g. `lease free --force --stop-job`).

Killing a process on Windows doesn't kill its children, so tools/batch.py touches a file every few seconds
and runs check it; if it's stale for a minute, the batch is gone and the run exits (train2018 resumes
from its last checkpoint next time).
"""

import os
import time
from pathlib import Path

ENV = "GVGAI_HEARTBEAT"
STALE_S = float(os.environ.get("GVGAI_HEARTBEAT_STALE", 60))


def check() -> None:
    p = os.environ.get(ENV)
    if p and time.time() - Path(p).stat().st_mtime > STALE_S:
        print("batch stopped: exiting", flush=True)
        os._exit(3)
