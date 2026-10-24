"""Pause training while Colin plays a game, and carry on after.

    uv run python -m tools.pause stop      # pause every training run now
    uv run python -m tools.pause go        # carry on
    uv run python -m tools.pause watch     # pause while a game runs, carry on a minute after it closes

A game is anything running from a steamapps/common or Epic Games folder, or League or Valorant.
A paused run keeps its GPU memory and its lease, and its train_wall_s includes the pause, so
don't quote speeds from a run that was paused. tools.batch's --max-minutes counts paused time too:
set it well above the run's own time (every frame + IMPALA alone takes about 3 hours).
"""

import argparse
import time

import psutil

TRAINERS = ("train.modern", "train.ddqn2018")
NOT_GAMES = ("wallpaper_engine", "aseprite")
RIOT = {"league of legends.exe", "valorant-win64-shipping.exe"}


def runs() -> list[psutil.Process]:
    """Every training process and its children."""
    out = {}
    for p in psutil.process_iter(["cmdline"]):
        try:
            if any(t in " ".join(p.info["cmdline"] or []) for t in TRAINERS):
                out |= {c.pid: c for c in (p, *p.children(recursive=True))}
        except psutil.NoSuchProcess:
            pass
    return list(out.values())


def game() -> str | None:
    for p in psutil.process_iter(["name", "exe"]):
        exe, name = (p.info["exe"] or "").lower(), (p.info["name"] or "").lower()
        if name in RIOT or (
            ("\\steamapps\\common\\" in exe or "\\epic games\\" in exe) and not any(n in exe for n in NOT_GAMES)
        ):
            return p.info["name"]
    return None


def each(action: str) -> int:
    n = 0
    for p in runs():
        try:
            getattr(p, action)()
            n += 1
        except psutil.NoSuchProcess:
            pass
    return n


def watch(grace_s: float) -> None:
    paused, last_game = False, 0.0
    while runs():
        g = game()
        if g:
            last_game = time.time()
            if not paused:
                print(f"{time.strftime('%H:%M')} {g} started: paused {each('suspend')} processes", flush=True)
                paused = True
        elif paused and time.time() - last_game > grace_s:
            print(f"{time.strftime('%H:%M')} no game for {grace_s:.0f} s: resumed {each('resume')} processes", flush=True)
            paused = False
        time.sleep(10)
    print(f"{time.strftime('%H:%M')} no training left, watch ends", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["stop", "go", "watch"])
    p.add_argument("--grace", type=float, default=60, help="seconds after a game closes before carrying on")
    a = p.parse_args()
    if a.what == "watch":
        watch(a.grace)
    else:
        print(f"{'paused' if a.what == 'stop' else 'resumed'} {each('suspend' if a.what == 'stop' else 'resume')} processes")
