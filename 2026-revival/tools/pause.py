"""Scale training down while Colin plays a game, and back up after.

    uv run python -m tools.pause stop      # pause every training run now
    uv run python -m tools.pause go        # carry on
    uv run python -m tools.pause watch     # while a game runs, keep --keep runs going at idle priority
                                           # and pause the rest; back to full a minute after it closes

A game is anything running from a steamapps/common or Epic Games folder, or League or Valorant.
A paused run keeps its GPU memory and its lease, and its train_wall_s includes the pause, so
don't quote speeds from a run that was paused. While anything is paused, runs/paused exists, so
tools.batch doesn't mistake a quiet run for a hung one.
"""

import argparse
import time

import psutil

from gvgai.games import ROOT

MARK = ROOT / "runs" / "paused"

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
    for _, p in sorted(tops.items()):
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
        except psutil.NoSuchProcess, psutil.AccessDenied:
            pass
    return n


def watch(grace_s: float, keep_n: int) -> None:
    paused, last_game = False, 0.0
    while runs():
        g = game()
        if g:
            last_game = time.time()
            if not paused:
                MARK.touch()
                each("resume")  # from a clean state, in case a watch before this one paused some
                gs = groups()
                keep, rest = gs[:keep_n], gs[keep_n:]
                each("nice", [p for r in keep for p in r], psutil.IDLE_PRIORITY_CLASS)
                each("suspend", [p for r in rest for p in r])
                print(
                    f"{time.strftime('%H:%M')} {g}: {len(keep)} runs at idle priority, {len(rest)} paused", flush=True
                )
                paused = True
        elif paused and time.time() - last_game > grace_s:
            each("nice", None, psutil.NORMAL_PRIORITY_CLASS)
            print(
                f"{time.strftime('%H:%M')} no game for {grace_s:.0f} s: all runs back ({each('resume')} processes)",
                flush=True,
            )
            MARK.unlink(missing_ok=True)
            paused = False
        time.sleep(10)
    MARK.unlink(missing_ok=True)
    print(f"{time.strftime('%H:%M')} no training left, watch ends", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["stop", "go", "watch"])
    p.add_argument("--grace", type=float, default=60, help="seconds after a game closes before carrying on")
    p.add_argument("--keep", type=int, default=2, help="runs that keep going (at idle priority) during a game")
    a = p.parse_args()
    if a.what == "watch":
        watch(a.grace, a.keep)
    else:
        stop = a.what == "stop"
        if stop:
            MARK.touch()
        else:
            MARK.unlink(missing_ok=True)
        print(f"{'paused' if stop else 'resumed'} {each('suspend' if stop else 'resume')} processes")
