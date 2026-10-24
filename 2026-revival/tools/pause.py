"""Scale training down while Colin plays a game, and back up after.

    uv run python -m tools.pause stop      # pause every training run now
    uv run python -m tools.pause go        # carry on
    uv run python -m tools.pause watch     # while a game runs, keep one run going at idle priority and
                                           # pause the rest; back to full a minute after it closes

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


def groups() -> list[list[psutil.Process]]:
    """One list per training run: its outermost training process and everything under it."""
    tops = {}
    for p in psutil.process_iter(["cmdline"]):
        try:
            if any(t in " ".join(p.info["cmdline"] or []) for t in TRAINERS):
                tops[p.pid] = p
        except psutil.NoSuchProcess:
            pass
    out = []
    for pid, p in sorted(tops.items()):
        try:
            if p.ppid() not in tops:
                out.append([p, *p.children(recursive=True)])
        except psutil.NoSuchProcess:
            pass
    return out


def runs() -> list[psutil.Process]:
    return [p for g in groups() for p in g]


def game() -> str | None:
    for p in psutil.process_iter(["name", "exe"]):
        exe, name = (p.info["exe"] or "").lower(), (p.info["name"] or "").lower()
        if name in RIOT or (
            ("\\steamapps\\common\\" in exe or "\\epic games\\" in exe) and not any(n in exe for n in NOT_GAMES)
        ):
            return p.info["name"]
    return None


def each(action: str, procs: list[psutil.Process] | None = None, *args) -> int:
    n = 0
    for p in runs() if procs is None else procs:
        try:
            getattr(p, action)(*args)
            n += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return n


def watch(grace_s: float) -> None:
    paused, last_game = False, 0.0
    while runs():
        g = game()
        if g:
            last_game = time.time()
            if not paused:
                keep, *rest = groups()
                each("nice", keep, psutil.IDLE_PRIORITY_CLASS)
                n = each("suspend", [p for r in rest for p in r])
                print(f"{time.strftime('%H:%M')} {g} started: 1 run at idle priority, {len(rest)} paused ({n} processes)", flush=True)
                paused = True
        elif paused and time.time() - last_game > grace_s:
            each("nice", None, psutil.NORMAL_PRIORITY_CLASS)
            print(f"{time.strftime('%H:%M')} no game for {grace_s:.0f} s: all runs back ({each('resume')} processes)", flush=True)
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
