"""The four games of the 2018 dissertation, and the levels every agent trains and is evaluated on."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # the 2026-revival folder: results/ and runs/ live here

GAMES = ["aliens", "chase", "deceptizelda", "waferthinmints2018"]

# Deceptizelda's level 1 is a bigger map than 2018's network could take, so every agent uses 0 and 2.
_LEVELS = {"deceptizelda": (0, 2)}
HELD_OUT = (3, 4)  # never trained on


def game_levels(game: str) -> tuple[int, ...]:
    return _LEVELS.get(game, (0, 1, 2))


def git_commit() -> str:
    """The code a run used, recorded in its result (read at the start: a commit made mid-run isn't it)."""
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
