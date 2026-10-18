"""Compile the engine into build/classes when any Java source changed. Run directly or via ensure_built()."""

import hashlib
import os
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
    """engine can be another copy of the engine (the golden test builds the unmodified import)."""
    src, out, stamp = engine / "src", engine / "build" / "classes", engine / "build" / "sources.sha256"
    files = sorted(src.rglob("*.java"))
    digest = _digest(src, files)
    if stamp.exists() and stamp.read_text() == digest:
        return out
    out.mkdir(parents=True, exist_ok=True)
    argfile = engine / "build" / "sources.txt"
    argfile.write_text("\n".join(f'"{f.as_posix()}"' for f in files))
    javac = JDK / "bin" / "javac"
    subprocess.run([str(javac), "-nowarn", "-encoding", "UTF-8", "-d", str(out), f"@{argfile}"], check=True)
    stamp.write_text(digest)
    return out


def java() -> str:
    return str(JDK / "bin" / "java")


if __name__ == "__main__":
    print(ensure_built())
