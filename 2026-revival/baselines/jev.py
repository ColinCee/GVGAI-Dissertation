"""TypeSafe's Jev picks every move from a short text description of the screen, with no training.

    uv run python -m baselines.jev --block 0 --streams 0-1     # a short batch: 2 of block 0's 8 game streams

Plays the agents' own test games (agents/evaluate.py: 8 streams of 13 games per --seed), in short batches
that can stop at any point: each game is saved to results/jev/<model>-s<block>.jsonl as it ends, a rerun
skips saved games, and no new game starts once --budget dollars are spent across everything saved.
The key is read from TYPESAFE_API_KEY (the environment, or the Windows user variables, so it works
without restarting the shell that started this). It is never printed or written anywhere.
Each game's moves, probabilities and timings go to runs/jev-<model>-s<block>/.
"""

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gvgai.env as genv
from gvgai.games import ROOT

URL = "https://api.typesafe.ai/v1/systemone"
PRICE_PER_M = 0.042  # $ per million input tokens, typesafe.ai pricing, Oct 2026
MOVES = {"shoot": 0, "left": 1, "right": 2}  # index into env.action_names: USE, LEFT, RIGHT
QUESTION = {
    "move": {
        "type": "choice",
        "instructions": (
            "You are playing Space Invaders. Aliens move sideways and drop bombs straight down. Win by destroying "
            "every alien, lose if a bomb or an alien touches your ship. Hitting an alien scores 2 points and hitting "
            "a shield scores 1 and destroys that piece of it. Only one of your shots can be in the air at a time. "
            "Pick the best move right now."
        ),
        "criteria": {
            "shoot": "fire a shot straight up from your ship's column",
            "left": "move two columns left",
            "right": "move two columns right",
        },
    }
}


# The wording block 0 (jev-latest, --warn) ran with, before shield hits were found to score 1 point.
QUESTION_V1 = {
    "move": {
        **QUESTION["move"],
        "instructions": (
            "You are playing Space Invaders. Aliens move sideways and drop bombs straight down. Win by destroying "
            "every alien, lose if a bomb or an alien touches your ship. Shots that hit a shield are wasted and "
            "only one of your shots can be in the air at a time. Pick the best move right now."
        ),
    }
}


def api_key() -> str:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            key = winreg.QueryValueEx(k, "TYPESAFE_API_KEY")[0]
    return key


def describe(sso, warn: bool = False) -> str:
    """The screen as text, in grid cells: column 0 is the left edge, row 0 the top."""
    b = sso.blockSize
    cols, rows = int(sso.worldDimension[0] // b), int(sso.worldDimension[1] // b)

    def cells(groups) -> list[tuple[int, int]]:
        return [
            (int(o.position.x // b) if hasattr(o.position, "x") else int(o.position[0] // b),
             int(o.position.y // b) if hasattr(o.position, "y") else int(o.position[1] // b))
            for row in groups or [] for o in row if o
        ]  # fmt: skip

    ax, ay = int(sso.avatarPosition[0] // b), int(sso.avatarPosition[1] // b)
    aliens = sorted(cells(sso.NPCPositions), key=lambda c: (abs(c[0] - ax), -c[1]))
    shields = sorted({c[0] for c in cells(sso.immovablePositions)})
    shots = cells(sso.fromAvatarSpritesPositions)
    bombs = sorted(cells(getattr(sso, "movablePositions", None)), key=lambda c: (abs(c[0] - ax), -c[1]))
    fmt = lambda cs: ", ".join(f"({x},{y})" for x, y in cs) or "none"  # noqa: E731
    return (
        f"Grid {cols} columns by {rows} rows, (column,row), row 0 at the top. "
        f"Your ship: ({ax},{ay}), it can only move left or right. "
        f"Your shot in the air: {fmt(shots)}{' (you cannot shoot until it lands)' if shots else ''}. "
        f"Aliens ({len(aliens)}), nearest first: {fmt(aliens[:8])}. "
        f"Bombs falling, nearest first: {fmt(bombs[:5])}. "
        f"Columns with a shield: {', '.join(map(str, shields)) or 'none'}."
    ) + (danger(bombs, ax, ay) if warn else "")


def rule(sso) -> int:
    """The --rule ablation: shoot, but step away from a bomb within 2 columns and 3 rows above the ship,
    using the same positions describe() gives Jev. Tells apart what the model adds from what the features do."""
    b = sso.blockSize
    ax, ay = int(sso.avatarPosition[0] // b), int(sso.avatarPosition[1] // b)
    cols = int(sso.worldDimension[0] // b)
    near = [int(o.position.x // b) for row in getattr(sso, "movablePositions", None) or [] for o in row
            if o and abs(int(o.position.x // b) - ax) <= 2 and 0 < ay - int(o.position.y // b) <= 3]  # fmt: skip
    if not near:
        return MOVES["shoot"]
    go_left = min(near, key=lambda x: abs(x - ax)) >= ax
    if (go_left and ax < 2) or (not go_left and ax > cols - 3):
        go_left = not go_left
    return MOVES["left" if go_left else "right"]


def describe_json(sso, warn: bool = False) -> dict:
    """--json: the same facts as describe(), as a JSON object (TypeSafe's docs: state can be text or JSON)."""
    b = sso.blockSize
    cell = lambda o: [int(o.position.x // b), int(o.position.y // b)]  # noqa: E731
    ax, ay = int(sso.avatarPosition[0] // b), int(sso.avatarPosition[1] // b)
    aliens = sorted((cell(o) for r in sso.NPCPositions or [] for o in r if o), key=lambda c: (abs(c[0] - ax), -c[1]))
    bombs = sorted((cell(o) for r in getattr(sso, "movablePositions", None) or [] for o in r if o),
                   key=lambda c: (abs(c[0] - ax), -c[1]))  # fmt: skip
    shots = [cell(o) for r in sso.fromAvatarSpritesPositions or [] for o in r if o]
    state = {
        "grid": {
            "columns": int(sso.worldDimension[0] // b),
            "rows": int(sso.worldDimension[1] // b),
            "coordinates": "[column, row], row 0 at the top",
        },  # fmt: skip
        "ship": {"at": [ax, ay], "moves": "left or right only"},
        "shot_in_air": shots[:1] or None,
        "can_shoot": not shots,
        "aliens_nearest_first": aliens[:8],
        "alien_count": len(aliens),
        "bombs_nearest_first": bombs[:5],
        "shield_columns": sorted({cell(o)[0] for r in sso.immovablePositions or [] for o in r if o}),
    }
    if warn:
        state["bombs_within_2_columns"] = [
            {"column": x, "from_you": x - ax, "rows_above": ay - y}
            for x, y in sorted((c for c in bombs if abs(c[0] - ax) <= 2 and c[1] < ay), key=lambda c: -c[1])
        ]
    return state


def danger(bombs: list[tuple[int, int]], ax: int, ay: int) -> str:
    """--warn: the bombs a move could run into (a move goes 2 columns), as facts, not advice."""
    near = sorted((b for b in bombs if abs(b[0] - ax) <= 2 and b[1] < ay), key=lambda b: -b[1])
    if not near:
        return " No bombs within 2 columns of your ship."
    parts = [f"column {x} ({x - ax:+d} from you), {ay - y} rows above" for x, y in near]
    return " Bombs within 2 columns of your ship: " + "; ".join(parts) + "."


def ask(state: str | dict, key: str, model: str, question: dict = QUESTION) -> tuple[dict, int, float]:
    body = json.dumps({"model": model, "state": state, "questions": question}).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(5):
        t = time.perf_counter()
        try:
            req = urllib.request.Request(URL, data=body, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read())
            return out["answers"]["move"], out["usage"]["input_tokens"], time.perf_counter() - t
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500:
                raise RuntimeError(f"Jev HTTP {e.code}: {e.read().decode()[:300]}") from None
        except urllib.error.URLError, TimeoutError:
            pass
        time.sleep(2**attempt)
    raise RuntimeError("Jev: 5 failed attempts")


def spent(results: Path) -> float:
    """Dollars spent by every game saved so far, across all blocks and models."""
    rows = [json.loads(ln) for f in results.glob("*.jsonl") for ln in f.read_text().splitlines() if ln]
    return sum(r["tokens"] for r in rows) * PRICE_PER_M / 1e6


def play_stream(block: int, stream: int, n_games: int, a, key: str, results: Path, lock: threading.Lock) -> None:
    """One evaluation game stream: the same engine seed, levels and game order as agents/evaluate.py's env
    `stream` for the run with --seed `block` (gvgai/vec.py), so Jev plays the agents' exact test games.
    A game already saved is replayed by holding shoot (free) to move the engine on to the next one."""
    seed = 7 + block
    path = results / f"{a.model}{a.tag}-s{block}.jsonl"
    saved = path.read_text().splitlines() if path.exists() else []
    done_games = {(r["stream"], r["game"]) for r in map(json.loads, saved)}
    if all((stream, g) in done_games for g in range(n_games)):
        return
    env = genv.GvgaiEnv("aliens", levels=(0, 1, 2), light_obs=False, engine_seed=seed * 1000 + stream)
    log_dir = ROOT / "runs" / f"jev-{a.model}{a.tag}-s{block}"
    log_dir.mkdir(parents=True, exist_ok=True)
    try:
        for g in range(n_games):
            env.reset(seed=seed + stream) if g == 0 else env.reset()
            skip = (stream, g) in done_games
            if not skip and spent(results) >= a.budget:
                print(f"budget ${a.budget} reached, stopping", flush=True)
                return
            done, info, calls, tokens, ms = False, {}, 0, 0, []
            log = None if skip else open(log_dir / f"stream{stream}-game{g}.jsonl", "w", encoding="utf-8")  # noqa: SIM115
            while not done:
                if skip:
                    move = MOVES["shoot"]
                elif a.rule:
                    move = rule(env.comm.sso)
                else:
                    state = (describe_json if a.json else describe)(env.comm.sso, a.warn)
                    ans, n, dt = ask(state, key, a.model, QUESTION_V1 if a.v1 else QUESTION)
                    calls, tokens, move = calls + 1, tokens + n, MOVES[ans["choice"]]
                    ms.append(dt * 1000)
                    log.write(json.dumps({"tick": env._ticks, "state": state, "move": ans["choice"],
                                          "p": ans["probabilities"], "ms": round(dt * 1000)}) + "\n")  # fmt: skip
                _, _, done, _, info = env.step(move, repeat=a.repeat)
            if skip:
                continue
            log.close()
            how = "win" if info["win"] else ("timeout" if info["timeout"] else "death")
            row = {"stream": stream, "game": g, "level": info["level"], "win": int(info["win"]), "score": info["score"],
                   "ticks": info["tick"], "how": how, "calls": calls, "tokens": tokens,
                   "ms_median": sorted(ms)[len(ms) // 2] if ms else 0, "ms_max": max(ms, default=0)}  # fmt: skip
            with lock:
                with open(path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(f"s{block} stream {stream} game {g} (level {info['level']}): {how}, {info['score']:.0f} points, "
                      f"${spent(results):.2f} spent", flush=True)  # fmt: skip
    finally:
        env.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--block", type=int, default=0, help="which agent evaluation's games to play (its --seed)")
    p.add_argument("--streams", default="0-7", help="which of its 8 game streams, e.g. 0-1 for a short batch")
    p.add_argument("--games", type=int, default=13, help="games per stream (evaluate plays ceil(100 / 8))")
    p.add_argument("--budget", type=float, default=4.5, help="stop before a new game once this many $ are spent")
    p.add_argument("--repeat", type=int, default=2, help="ticks each move is held, as the final agent")
    p.add_argument("--model", default="jev-latest")
    p.add_argument("--warn", action="store_true", help="also list the bombs within 2 columns of the ship")
    p.add_argument("--json", action="store_true", help="the state as a JSON object instead of sentences")
    p.add_argument("--v1", action="store_true", help="block 0's question wording (QUESTION_V1), to compare with it")
    p.add_argument("--rule", action="store_true", help="no Jev: rule() on the same features (free, an ablation)")
    a = p.parse_args()
    a.tag = ("-warn" if a.warn else "") + ("-json" if a.json else "") + ("-v1" if a.v1 else "")
    if a.rule:
        a.model, a.tag = "rule", ""

    key = api_key()
    genv.ClientCommGYM.ClientCommGYM.SSO_TYPE = "BOTH"  # sprite positions, not just the frame
    results = ROOT / "results" / "jev"
    results.mkdir(parents=True, exist_ok=True)
    lo, _, hi = a.streams.partition("-")
    lock = threading.Lock()
    with ThreadPoolExecutor() as pool:
        streams = range(int(lo), int(hi or lo) + 1)
        jobs = [pool.submit(play_stream, a.block, s, a.games, a, key, results, lock) for s in streams]
        for j in jobs:
            j.result()
    path = results / f"{a.model}{a.tag}-s{a.block}.jsonl"
    rows = [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []
    if rows:
        wins, pts = sum(r["win"] for r in rows), sum(r["score"] for r in rows) / len(rows)
        print(f"block {a.block}: {len(rows)} games, {wins / len(rows):.0%} won, {pts:.1f} points; "
              f"${spent(results):.2f} spent in all")  # fmt: skip


if __name__ == "__main__":
    main()
