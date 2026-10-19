"""Build the reference engine for the golden test: the unmodified import plus only the seed switch.

uv run python tests/pristine_engine.py      # prints the folder; pass it to golden.py record --engine
"""

import io
import shutil
import subprocess
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
REPO = HERE.parent
IMPORT_COMMIT = "16968e4"  # "Import GVGAI_GYM engine ... unmodified"; this folder was called revival/ then
OUT = HERE / "runs" / "pristine-engine"


def main() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    tar = subprocess.run(["git", "archive", IMPORT_COMMIT, "revival/engine/src", "revival/engine/sprites"],
                         cwd=REPO, capture_output=True, check=True).stdout  # fmt: skip
    with tarfile.open(fileobj=io.BytesIO(tar)) as t:
        t.extractall(OUT.parent / "_pristine_tmp", filter="data")
    shutil.move(OUT.parent / "_pristine_tmp" / "revival" / "engine", OUT)
    shutil.rmtree(OUT.parent / "_pristine_tmp")
    shutil.copytree(HERE / "engine" / "games", OUT / "games")  # data, includes waferthinmints2018
    # The seed switch is LearningMachine.java's only change since the import; take that file as it is now.
    now = HERE.relative_to(REPO).as_posix()
    diff = ["git", "diff", "--name-only", f"{IMPORT_COMMIT}:revival/engine/src/tracks", f"HEAD:{now}/engine/src/tracks"]
    changed = subprocess.run(diff, cwd=REPO, capture_output=True, check=True, text=True).stdout.split()
    assert "LearningMachine.java" in changed, changed  # the other changes are the speed work it must not have
    shutil.copy(
        HERE / "engine" / "src" / "tracks" / "LearningMachine.java", OUT / "src" / "tracks" / "LearningMachine.java"
    )
    print(OUT)


if __name__ == "__main__":
    main()
