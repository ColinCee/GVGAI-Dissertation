"""Compile the engine into build/classes when any Java source changed. Run directly or via ensure_built()."""

import hashlib
import os
import subprocess
from pathlib import Path

ENGINE = Path(__file__).resolve().parent
SRC = ENGINE / "src"
OUT = ENGINE / "build" / "classes"
STAMP = ENGINE / "build" / "sources.sha256"
JDK = Path(os.environ.get("GVGAI_JDK", r"D:\Java\jdk-25.0.4.1+1"))


def _sources() -> list[Path]:
    return sorted(SRC.rglob("*.java"))


def _digest(files: list[Path]) -> str:
    h = hashlib.sha256()
    for f in files:
        h.update(str(f.relative_to(SRC)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def ensure_built() -> Path:
    files = _sources()
    digest = _digest(files)
    if STAMP.exists() and STAMP.read_text() == digest:
        return OUT
    OUT.mkdir(parents=True, exist_ok=True)
    argfile = ENGINE / "build" / "sources.txt"
    argfile.write_text("\n".join(f'"{f.as_posix()}"' for f in files))
    javac = JDK / "bin" / "javac"
    subprocess.run([str(javac), "-nowarn", "-encoding", "UTF-8", "-d", str(OUT), f"@{argfile}"], check=True)
    STAMP.write_text(digest)
    return OUT


def java() -> str:
    return str(JDK / "bin" / "java")


if __name__ == "__main__":
    print(ensure_built())
