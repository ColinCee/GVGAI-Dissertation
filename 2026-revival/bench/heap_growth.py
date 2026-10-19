"""One Aliens game server under random play for 20 s: ticks/s and the Java process's memory, for JVM options.

uv run python bench/heap_growth.py -Xmx256m -XX:+UseSerialGC
"""

import random
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgai.env import GvgaiEnv  # noqa: E402

if __name__ == "__main__":
    opts = sys.argv[1:]
    env = GvgaiEnv("aliens", engine_seed=1, java_opts=opts)
    env.reset()
    t0, n = time.perf_counter(), 0
    while time.perf_counter() - t0 < 20:
        *_, term, _, _ = env.step(random.randrange(3))
        n += 1
        if term:
            env.reset()
    ps = f"(Get-Process -Id {env.comm.java.pid}).WorkingSet64/1MB"
    mb = float(subprocess.run(["powershell", "-c", ps], capture_output=True, text=True).stdout.strip())
    print(opts, f"{n / 20:.0f} ticks/s", f"{mb:.0f} MB")
    env.close()
