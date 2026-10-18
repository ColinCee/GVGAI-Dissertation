"""Golden test: the engine must replay recorded games exactly.

Each game is played with a fixed engine seed and fixed scripted moves on levels 0-2. Per tick we keep
a hash of the game state the server reports and of the decoded pixels (not the PNG bytes, which can
change between JDKs while the image stays the same). Speed changes must leave these identical.

    uv run python tests/golden.py record --engine "$(uv run python tests/pristine_engine.py)"
    uv run python tests/golden.py check  [--engine DIR]   # compare, exit 1 on any difference
"""

import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gvgai.env import ENGINE_DIR, GvgaiEnv  # noqa: E402

GAMES = ["aliens", "chase", "deceptizelda", "waferthinmints2018"]
LEVELS = [0, 1, 2]
ENGINE_SEED = 20180417  # date of the last 2018 commit
MAX_TICKS = 600  # per game; enough to cover deaths, wins and every sprite type in these levels
OUT = Path(__file__).resolve().parent / "golden"
# All the server sends in image mode; sprite positions are covered by the pixel hash.
STATE_FIELDS = ["gameScore", "gameTick", "gameWinner", "isGameOver", "availableActions"]


def _h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def play(game: str, engine_dir: str, engine_seed: int = ENGINE_SEED) -> list[dict]:
    env = GvgaiEnv(game, levels=LEVELS, max_ticks=MAX_TICKS, engine_seed=engine_seed, engine_dir=engine_dir)
    moves = random.Random(f"{ENGINE_SEED}-{game}")
    episodes = []
    try:
        for level in LEVELS:
            obs, _ = env.reset(options={"level": level})
            ticks = [_h(obs.tobytes())]
            while True:
                obs, _, term, trunc, info = env.step(moves.randrange(env.n_actions))
                sso = env.comm.sso
                state = json.dumps([getattr(sso, f, None) for f in STATE_FIELDS], sort_keys=True, default=str)
                ticks.append(f"{_h(state.encode())}{_h(obs.tobytes())}")
                if term or trunc:
                    break
            episodes.append(
                {"level": level, "score": info["score"], "winner": info["winner"], "ticks": ticks}
            )
    finally:
        env.close()
    return episodes


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["record", "check"])
    p.add_argument("--engine", default=ENGINE_DIR, help="record from tests/pristine_engine.py's folder")
    p.add_argument("--games", nargs="+", default=GAMES)
    p.add_argument("--engine-seed", type=int, default=ENGINE_SEED, help="change only to test the test")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    failed = False
    for game in a.games:
        got = play(game, a.engine, a.engine_seed)
        path = OUT / f"{game}.json"
        if a.mode == "record":
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
            source = "unmodified import + seed switch" if a.engine != ENGINE_DIR else "current engine"
            path.write_text(json.dumps({"engine": source, "commit": commit.stdout.strip(), "episodes": got}))
            print(f"{game}: recorded {sum(len(e['ticks']) for e in got)} ticks")
            continue
        want = json.loads(path.read_text())["episodes"]
        for ep_want, ep_got in zip(want, got, strict=True):
            for t, (w, g) in enumerate(zip(ep_want["ticks"], ep_got["ticks"], strict=False)):
                if w != g:
                    what = "state" if w[:12] != g[:12] else "pixels"
                    print(f"{game} level {ep_want['level']}: {what} differ first at tick {t}")
                    failed = True
                    break
            else:
                if len(ep_want["ticks"]) != len(ep_got["ticks"]):
                    print(
                        f"{game} level {ep_want['level']}: "
                        f"length {len(ep_got['ticks'])} != {len(ep_want['ticks'])}"
                    )
                    failed = True
        if not failed:
            print(f"{game}: identical")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
