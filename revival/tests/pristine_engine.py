"""Build the reference engine for the golden test: the unmodified import plus only the seed switch.

uv run python tests/pristine_engine.py      # prints the folder; pass it to golden.py record --engine
"""

import io
import shutil
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IMPORT_COMMIT = "16968e4"  # "Import GVGAI_GYM engine ... unmodified"
OUT = REPO / "revival" / "runs" / "pristine-engine"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    tar = subprocess.run(["git", "archive", IMPORT_COMMIT, "revival/engine/src", "revival/engine/sprites"],
                         cwd=REPO, capture_output=True, check=True).stdout  # fmt: skip
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(OUT.parent / "_pristine_tmp", filter="data")
    shutil.move(OUT.parent / "_pristine_tmp" / "revival" / "engine", OUT)
    shutil.rmtree(OUT.parent / "_pristine_tmp")
    shutil.copytree(REPO / "revival" / "engine" / "games", OUT / "games")  # data, includes waferthinmints2018
    # The seed switch is LearningMachine.java's only change since the import; take that file as it is now.
    diff = ["git", "diff", "--name-only", IMPORT_COMMIT, "HEAD", "--", "revival/engine/src/tracks"]
    changed = subprocess.run(diff, cwd=REPO, capture_output=True, check=True, text=True).stdout.split()
    assert changed == ["revival/engine/src/tracks/LearningMachine.java"], changed
    shutil.copy(REPO / changed[0], OUT / "src" / "tracks" / "LearningMachine.java")
    print(OUT)


if __name__ == "__main__":
    main()
