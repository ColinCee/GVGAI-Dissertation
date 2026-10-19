"""Compile the engine into build/classes-<source hash>. Run directly or via ensure_built()."""

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
JDK = Path(os.environ.get("GVGAI_JDK", r"D:\Java\jdk-25.0.4.1+1"))


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
    javac = JDK / "bin" / "javac"
    subprocess.run([str(javac), "-nowarn", "-encoding", "UTF-8", "-d", str(tmp), f"@{argfile}"], check=True)
    (tmp / "ok").write_text("")
    try:
        tmp.rename(out)
    except OSError:  # another process built the same sources first
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def java() -> str:
    return str(JDK / "bin" / "java")


if __name__ == "__main__":
    print(ensure_built())
