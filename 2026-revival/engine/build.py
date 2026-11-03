"""Compile the engine into build/classes-<source hash>. Run directly or via ensure_built()."""

import hashlib
import os
import shutil
import subprocess
import sys
from functools import cache
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
DEFAULT_JDK = Path(r"D:\Java\jdk-25.0.4.1+1")


@cache
def jdk() -> Path:
    """The JDK 25 folder: GVGAI_JDK, else the default folder, else JAVA_HOME, else the `java` on PATH."""
    if env := os.environ.get("GVGAI_JDK"):
        return Path(env)
    if DEFAULT_JDK.exists():
        return DEFAULT_JDK
    if home := os.environ.get("JAVA_HOME"):
        return Path(home)
    if on_path := shutil.which("java"):
        return Path(on_path).resolve().parent.parent
    sys.exit("No JDK found: the engine needs JDK 25. Install it and set GVGAI_JDK or JAVA_HOME to its folder.")


def _digest(src: Path, files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(src)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def ensure_built(engine: Path = ENGINE) -> Path:
    """Compile into build/classes-<source hash> unless that exists. A new build never touches the
    folder running servers use (they load classes lazily), so it is safe while runs are going.
    `engine` can be another copy of the engine (the golden test builds the unmodified import)."""
    src = engine / "src"
    files = sorted(src.rglob("*.java"))
    out = engine / "build" / f"classes-{_digest(src, files)[:12]}"
    if (out / "ok").exists():
        return out
    tmp = out.with_name(f"{out.name}.tmp{os.getpid()}")  # per process: several runs may build at once
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    argfile = tmp / "sources.txt"
    argfile.write_text("\n".join(f'"{f.as_posix()}"' for f in files))
    javac = jdk() / "bin" / "javac"
    subprocess.run([str(javac), "-nowarn", "-encoding", "UTF-8", "-d", str(tmp), f"@{argfile}"], check=True)
    (tmp / "ok").write_text("")
    try:
        tmp.rename(out)
    except OSError:  # another process built the same sources first
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def java() -> str:
    return str(jdk() / "bin" / "java")


if __name__ == "__main__":
    print(ensure_built())
