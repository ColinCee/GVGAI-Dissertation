"""Record one greedy game as video, for showing agents play: what the viewer sees, one engine tick per frame.

    uv run python -m tools.record --agent ddqn2018:weights/aliens-ddqn2018-s0-agent.pt --game aliens --level 0 \
        --seed 3 --out clips --name aliens-2018-agent

Agents (each plays exactly as its evaluation does; only the frames kept for the viewer differ):
- `ddqn2018:<weights>`: the 2018 agent (agents/ddqn2018.py's play_episode, weights from the last target sync);
- `modern:<weights>:<preset>`: a train.modern network; greedy, the action held 4 ticks, a stack of 4 gray-half
  frames (computed here from the colour frame exactly as the engine's gvgai.grayHalf does);
- `hold:<action>` (e.g. hold:use) and `random[:<rng seed>]`: baselines/trivial.py's players, 4 ticks per move;
- `jev[:<model>[:warn]]`: baselines/jev.py's player (TypeSafe's Jev on a text state; needs TYPESAFE_API_KEY);
- `rule`: baselines/jev.py's --rule (shoot, step away from bombs overhead; no model);
- `mcts:<iterations>[:<search seed>]`: the engine's sampleMCTS with a fixed search budget, played inside the
  engine (it needs the forward model); its frames come from the engine itself.

`--seed` is the engine seed (GvgaiEnv's engine_seed): the same level and seed give every agent the same game.
Writes <name>.mp4, <name>.webm, <name>.png (poster) and <name>.json (result and every decision, with the
Q-values of learning agents). Other modes: `--still` (the start frame and the level's tile positions) and
`--features <impala weights>` (each conv layer's activations for one frame stack, for a network diagram).
CPU only.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from agents.ddqn2018 import QNet, play_episode, preprocess
from agents.modern import STACK, Net
from gvgai.env import GvgaiEnv
from gvgai.games import ROOT
from train.modern import PRESETS

sys.path.insert(0, str(ROOT / "engine"))
import build  # noqa: E402

SCALE = 2
VIEWER_TICKS_PER_S_2018 = 40  # the 2018 engine slept 25 ms per tick for human viewers


def gray_half(rgb: np.ndarray) -> np.ndarray:
    """The engine's gvgai.grayHalf (VGDLViewer.saveImage): PIL's "L" weights, then 2x2 averages."""
    c = rgb.astype(np.int64)
    luma = (c[..., 0] * 19595 + c[..., 1] * 38470 + c[..., 2] * 7471 + 0x8000) >> 16
    h, w = luma.shape[0] // 2 * 2, luma.shape[1] // 2 * 2
    s = luma[0:h:2, 0:w:2] + luma[0:h:2, 1:w:2] + luma[1:h:2, 0:w:2] + luma[1:h:2, 1:w:2]
    return ((s + 2) >> 2).astype(np.uint8)


def java_ints(seed: int, n: int) -> list[int]:
    """java.util.Random(seed).nextInt() n times: the per-level seeds the server draws with -Dgvgai.seed."""
    mask = (1 << 48) - 1
    s, out = (seed ^ 0x5DEECE66D) & mask, []
    for _ in range(n):
        s = (s * 0x5DEECE66D + 0xB) & mask
        v = s >> 16
        out.append(v - (1 << 32) if v >= 1 << 31 else v)
    return out


# The server draws a level seed for each level it plays: two while connecting, then one per reset.
# Checked by replaying an engine-side game in GvgaiEnv frame for frame (--verify on an mcts agent).
FIRST_GAME_SEED_INDEX = 2


def level_seed(engine_seed: int) -> int:
    return java_ints(engine_seed, FIRST_GAME_SEED_INDEX + 1)[FIRST_GAME_SEED_INDEX]


class Recorder:
    """GvgaiEnv with every frame kept, for play_episode: frames[i] is the screen after i ticks."""

    def __init__(self, env: GvgaiEnv):
        self.env, self.frames, self.info = env, [], {}

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.frames = [obs.copy()]
        return obs, info

    def step(self, action, repeat=1):
        assert repeat == 1
        obs, r, term, trunc, info = self.env.step(action)
        self.frames.append(obs.copy())
        assert info["tick"] == len(self.frames) - 1
        self.info = info
        return obs, r, term, trunc, info


# --- players --------------------------------------------------------------------------------------


def load_2018(path: str, h: int, w: int, n_actions: int) -> QNet:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = QNet(h, w, n_actions, dueling=False)
    # evaluation used the weights from the last target sync (train/ddqn2018.py)
    net.load_state_dict(ck["synced"] if ck.get("synced") is not None else ck["q"])
    return net.eval()


def load_modern(path: str, preset: str, h: int, w: int, n_actions: int) -> Net:
    cfg = PRESETS[preset]()
    cfg.compile = False
    net = Net(h, w, n_actions, cfg)
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.eval()


def play_2018(spec: str, game: str, level: int, seed: int) -> dict:
    env = Recorder(GvgaiEnv(game, levels=(level,), include_nil=True, engine_seed=seed))
    names = env.action_names[1:]  # agent action i is env action i + 1 (0 is NIL)
    h, w = preprocess(env.comm.sso.image).shape
    net = load_2018(spec, h, w, len(names))
    decisions: list[dict] = []
    stacks: list[np.ndarray] = []

    class Shim:  # what play_episode reads from the agent when not training
        class cfg:
            warmup_stacks = 0

        total_stacks, epsilon = 0, 0.0

        @staticmethod
        @torch.no_grad()
        def act(stack, greedy, net):
            q = net(torch.from_numpy(stack[None]))[0]
            a = int(q.argmax().item())
            decisions.append({"tick": len(env.frames) - 1, "action": names[a], "score": env.comm.sso.gameScore,
                              "q": [round(float(v), 4) for v in q]})  # fmt: skip
            stacks.append(stack)
            return a

    row = play_episode(env, Shim, train=False, greedy=True, net=net, level=level)
    frames = env.frames
    env.env.close()
    return {"frames": frames, "decisions": decisions, "names": names, "stacks": stacks,
            "win": bool(row["win"]), "score": float(row["score"]), "ticks": env.info["tick"],
            "agent_notes": "2018 agent: greyscale + Lanczos half-size frames, stacks of 4 consecutive ticks, a "
                           "decision every 4 ticks; weights from the last target sync, as its evaluation"}  # fmt: skip


def play_repeat(policy, names: list[str], env: GvgaiEnv, level: int, q_net=None, repeat: int = 4) -> dict:
    """modern / hold / random: the evaluation's loop (agents/evaluate.py) one tick at a time. A stack starts as
    the first frame repeated; each decision holds its action `repeat` ticks (the server's repeat stops at game end,
    tests/repeat_check.py shows the two are identical)."""
    obs, _ = env.reset(options={"level": level})
    frames = [obs.copy()]
    stack = deque([gray_half(obs)] * STACK, maxlen=STACK)
    decisions: list[dict] = []
    info: dict = {"tick": 0, "score": 0.0, "win": False}
    done = False
    while not done:
        x = np.stack(stack)[None]
        q = None
        if q_net is not None:
            with torch.no_grad():
                q = q_net.q(torch.from_numpy(x))[0]
            a = int(q.argmax().item())
        else:
            a = policy()
        d = {"tick": len(frames) - 1, "action": names[a], "score": env.comm.sso.gameScore}
        if q is not None:
            d["q"] = [round(float(v), 4) for v in q]
        decisions.append(d)
        for _ in range(repeat):
            obs, _, term, trunc, info = env.step(a)
            frames.append(obs.copy())
            if term or trunc:
                done = True
                break
        stack.append(gray_half(obs))
    env.close()
    return {"frames": frames, "decisions": decisions, "names": names, "win": bool(info["win"]),
            "score": float(info["score"]), "ticks": info["tick"]}  # fmt: skip


def play_modern(path: str, preset: str, game: str, level: int, seed: int) -> dict:
    env = GvgaiEnv(game, levels=(level,), engine_seed=seed)
    h, w = gray_half(env.comm.sso.image).shape
    net = load_modern(path, preset, h, w, env.n_actions)
    repeat = PRESETS[preset]().action_repeat
    out = play_repeat(None, list(env.action_names), env, level, q_net=net, repeat=repeat)
    out["agent_notes"] = (f"modern DQN preset {preset}: gray-half frames, stack of 4 decision frames, action "
                          f"held {repeat} ticks; fp32 on the CPU (its evaluation ran bf16 on the GPU, so a near-tie "
                          "can go the other way)")  # fmt: skip
    return out


def hold_index(names: list[str], action: str) -> int:
    want = action.upper() if action.upper().startswith("ACTION_") else "ACTION_" + action.upper()
    if want not in names:
        raise SystemExit(f"{action}: not one of {names}")
    return names.index(want)


def play_trivial(kind: str, arg: str, game: str, level: int, seed: int) -> dict:
    env = GvgaiEnv(game, levels=(level,), engine_seed=seed)
    names = list(env.action_names)
    if kind == "hold":
        a = hold_index(names, arg)
        out = play_repeat(lambda: a, names, env, level)
        out["agent_notes"] = f"{names[a]} held all game"
    else:
        rng = np.random.default_rng(int(arg or 0))
        out = play_repeat(lambda: int(rng.integers(0, len(names))), names, env, level)
        out["agent_notes"] = f"uniform random move every 4 ticks, numpy default_rng({int(arg or 0)})"
    return out


def play_jev(model: str, warn: bool, game: str, level: int, seed: int) -> dict:
    """baselines/jev.py's player: Jev picks each move from the text state, held 2 ticks as in its evaluation."""
    import gvgai.env as genv
    from baselines import jev

    key = jev.api_key()
    genv.ClientCommGYM.ClientCommGYM.SSO_TYPE = "BOTH"
    env = GvgaiEnv(game, levels=(level,), light_obs=False, engine_seed=seed)
    probs: list[dict] = []

    def policy() -> int:
        ans, _, _ = jev.ask(jev.describe(env.comm.sso, warn), key, model)
        probs.append(ans["probabilities"])
        return jev.MOVES[ans["choice"]]

    out = play_repeat(policy, list(env.action_names), env, level, repeat=2)
    for d, pr in zip(out["decisions"], probs, strict=True):
        d["p"] = pr
    out["agent_notes"] = (f"TypeSafe {model} on baselines/jev.py's text state{' with --warn' if warn else ''}, a move "
                          "every 2 ticks, no training")  # fmt: skip
    return out


def play_rule(game: str, level: int, seed: int) -> dict:
    """baselines/jev.py's --rule: shoot, step away from a bomb within 2 columns and 3 rows above; a move
    every 2 ticks."""
    import gvgai.env as genv
    from baselines import jev

    genv.ClientCommGYM.ClientCommGYM.SSO_TYPE = "BOTH"
    env = GvgaiEnv(game, levels=(level,), light_obs=False, engine_seed=seed)
    out = play_repeat(lambda: jev.rule(env.comm.sso), list(env.action_names), env, level, repeat=2)
    out["agent_notes"] = "baselines/jev.py rule(): shoot, step away from bombs overhead, a move every 2 ticks"
    return out


JAVA_MCTS = r"""
import core.game.Game;
import core.game.StateObservation;
import core.player.AbstractPlayer;
import core.vgdl.VGDLFactory;
import core.vgdl.VGDLParser;
import core.vgdl.VGDLRegistry;
import core.vgdl.VGDLViewer;
import java.io.*;
import java.util.Random;
import ontology.Types;
import tools.ElapsedCpuTimer;
import tracks.singlePlayer.advanced.sampleMCTS.SingleMCTSPlayer;

// RecordMCTS game_file level_file game_seed out_prefix search_seed: sampleMCTS (-Dgvgai.mctsIters per move)
// played as the learning server plays a game (its 10-pixel tiles, nothing chosen on tick 0), writing every
// frame drawn (raw RGB) and one "tick action score" line per move.
public class RecordMCTS {
    static OutputStream frames;
    static void frame() {
        try {
            if (VGDLViewer.lastRgb != null)
                frames.write(VGDLViewer.lastRgb, 0, VGDLViewer.lastWidth * VGDLViewer.lastHeight * 3);
        } catch (IOException e) { throw new UncheckedIOException(e); }
    }
    public static void main(String[] a) throws Exception {
        core.competition.CompetitionParameters.IS_LEARNING = true;
        VGDLFactory.GetInstance().init();
        VGDLRegistry.GetInstance().init();
        Game g = new VGDLParser().parseGame(a[0]);
        int seed = Integer.parseInt(a[2]);
        long searchSeed = Long.parseLong(a[4]);
        g.buildLevel(a[1], seed);
        frames = new BufferedOutputStream(new FileOutputStream(a[3] + ".rgb"), 1 << 20);
        PrintWriter log = new PrintWriter(new FileWriter(a[3] + ".log"));
        StateObservation start = g.getObservation();
        StringBuilder names = new StringBuilder("ACTIONS");
        for (Types.ACTIONS act : start.getAvailableActions()) names.append(" ").append(act);
        AbstractPlayer mcts = new tracks.singlePlayer.advanced.sampleMCTS.Agent(start, new ElapsedCpuTimer()) {
            public SingleMCTSPlayer getPlayer(StateObservation so, ElapsedCpuTimer t) {
                return new SingleMCTSPlayer(new Random(searchSeed), num_actions, actions);
            }
        };
        log.println(names);
        AbstractPlayer p = new AbstractPlayer() {
            public Types.ACTIONS act(StateObservation so, ElapsedCpuTimer t) {
                frame();  // drawn after the previous tick
                Types.ACTIONS act = so.getGameTick() == 0 ? Types.ACTIONS.ACTION_NIL : mcts.act(so, t);
                log.println(so.getGameTick() + " " + act + " " + so.getGameScore());
                return act;
            }
        };
        p.setup(null, seed, false);
        g.playOnlineGame(new AbstractPlayer[] {p}, seed, false, 0);
        frame();
        g.handleResult();
        double[] r = g.getFullResult();
        log.println("RESULT " + r[0] + " " + r[1] + " " + r[2] + " "
                + VGDLViewer.lastWidth + " " + VGDLViewer.lastHeight);
        log.close();
        frames.close();
    }
}
"""


def play_mcts(iters: int, search_seed: int, game: str, level: int, seed: int) -> dict:
    gdir = ROOT / "engine" / "games" / f"{game}_v0"
    classes = build.ensure_built()
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "RecordMCTS.java"
        src.write_text(JAVA_MCTS)
        cmd = [build.java(), "-XX:+UseSerialGC", "-Xmx512m", "-Dgvgai.rawFrames=true", "-Dgvgai.longDelay=0",
               "-Dgvgai.actionTime=100000000", f"-Dgvgai.mctsIters={iters}", "-Dgvgai.maxTicks=2000",
               "-cp", str(classes), str(src), str(gdir / f"{game}.txt"), str(gdir / f"{game}_lvl{level}.txt"),
               str(level_seed(seed)), str(Path(tmp) / "out"), str(search_seed)]  # fmt: skip
        # cwd: the engine loads sprites by relative path
        subprocess.run(cmd, cwd=ROOT / "engine", check=True, stdout=subprocess.DEVNULL, timeout=3600)
        lines = (Path(tmp) / "out.log").read_text().splitlines()
        _, win, score, _, w, h = lines[-1].split()
        raw = np.fromfile(Path(tmp) / "out.rgb", np.uint8)
    frames = list(raw.reshape(-1, int(h), int(w), 3))
    names = lines[0].split()[1:]  # the game's actions, in the engine's order (as GvgaiEnv's)
    moves = [ln.split() for ln in lines[1:-1]]
    decisions = [{"tick": int(t), "action": a, "score": float(s)} for t, a, s in moves]
    # the game's tick counter reads 1 on the first frame the client sees (as GvgaiEnv's reset); align
    for d in decisions:
        d["tick"] -= 1
    decisions = decisions[1:]  # tick 0 is played before any agent sees the game, as in GvgaiEnv
    return {"frames": frames, "decisions": decisions, "names": names, "win": float(win) > 0,
            "score": float(score), "ticks": len(frames) - 1,
            "agent_notes": f"engine's sampleMCTS, {iters} iterations per move, a move every tick, search "
                           f"Random({search_seed}); game played in the engine (game seed {level_seed(seed)}, "
                           "the one GvgaiEnv draws for this engine seed)"}  # fmt: skip


# --- checks: does the recorded game match the evaluation's own loop? ------------------------------------


def verify(kind: str, parts: list[str], game: str, level: int, seed: int, rec: dict) -> str:
    if kind == "ddqn2018":
        env = GvgaiEnv(game, levels=(level,), include_nil=True, engine_seed=seed)
        h, w = preprocess(env.comm.sso.image).shape
        net = load_2018(parts[1], h, w, env.n_actions - 1)

        class A:
            class cfg:
                warmup_stacks = 0

            total_stacks, epsilon = 0, 0.0

            @staticmethod
            @torch.no_grad()
            def act(stack, greedy, net):
                return int(net(torch.from_numpy(stack[None])).argmax(1).item())

        row = play_episode(env, A, train=False, greedy=True, net=net, level=level)
        env.close()
        got = (bool(row["win"]), float(row["score"]), row["ticks"])
    elif kind == "mcts":  # replay the engine-side game's moves through GvgaiEnv, frame for frame
        env = GvgaiEnv(game, levels=(level,), include_nil=True, engine_seed=seed)
        obs, _ = env.reset(options={"level": level})
        same = np.array_equal(obs, rec["frames"][0])
        info: dict = {}
        for d in rec["decisions"]:
            obs, _, term, trunc, info = env.step(env.action_names.index(d["action"]))
            same &= np.array_equal(obs, rec["frames"][d["tick"] + 1])
            if term or trunc:
                break
        env.close()
        got = (bool(info["win"]), float(info["score"]), info["tick"])
        if not same:
            return f"MISMATCH: frames differ in GvgaiEnv replay; replay result {got}"
    else:  # the evaluation's own setup: gray-half frames from the server, server-side repeat of 4
        env = GvgaiEnv(game, levels=(level,), engine_seed=seed, gray_half=True)
        obs, _ = env.reset(options={"level": level})
        names = list(env.action_names)
        net = load_modern(parts[1], parts[2], *obs.shape, len(names)) if kind == "modern" else None
        rng = np.random.default_rng(int(parts[1]) if kind == "random" and len(parts) > 1 and parts[1] else 0)
        stack = deque([obs] * STACK, maxlen=STACK)
        while True:
            if net is not None:
                with torch.no_grad():
                    a = int(net.q(torch.from_numpy(np.stack(stack)[None])).argmax(1).item())
            elif kind == "hold":
                a = hold_index(names, parts[1])
            else:
                a = int(rng.integers(0, len(names)))
            obs, _, term, trunc, info = env.step(a, 4)
            stack.append(obs)
            if term or trunc:
                break
        env.close()
        got = (bool(info["win"]), float(info["score"]), info["tick"])
    want = (rec["win"], rec["score"], rec["ticks"])
    return "ok" if got == want else f"MISMATCH: evaluation loop gave {got}, recording {want}"


# --- output ---------------------------------------------------------------------------------------------


def upscale(frame: np.ndarray) -> np.ndarray:
    big = frame.repeat(SCALE, 0).repeat(SCALE, 1)
    h, w = big.shape[:2]
    return np.pad(big, ((0, h % 2), (0, w % 2), (0, 0)), mode="edge")  # yuv420p needs even sides


def encode(frames: list[np.ndarray], path: Path, fps: float, codec: list[str]) -> None:
    h, w = upscale(frames[0]).shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", *codec, str(path)]  # fmt: skip
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    assert p.stdin is not None
    for f in frames:
        p.stdin.write(upscale(f).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise SystemExit(f"ffmpeg failed on {path}")


def write_clip(rec: dict, out: Path, name: str, fps: float, max_ticks: int | None, webm_crf: int) -> dict:
    frames = rec["frames"][: max_ticks + 1] if max_ticks else rec["frames"]
    encode(frames, out / f"{name}.mp4", fps, ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "28",
                                              "-preset", "veryslow", "-movflags", "+faststart"])  # fmt: skip
    encode(frames, out / f"{name}.webm", fps, ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-b:v", "0",
                                               "-crf", str(webm_crf), "-row-mt", "1", "-deadline", "good"])  # fmt: skip
    poster = len(frames) // 2
    Image.fromarray(upscale(frames[poster])).save(out / f"{name}.png", optimize=True)
    h, w = upscale(frames[0]).shape[:2]
    return {"video_frames": len(frames), "video_ticks": len(frames) - 1, "trimmed": len(frames) < len(rec["frames"]),
            "fps": fps, "ticks_per_frame": 1, "ticks_per_second": fps,
            "speed": f"{fps / VIEWER_TICKS_PER_S_2018:.2f}x the 2018 engine's viewing pace (40 ticks/s)",
            "scale": SCALE, "width": w, "height": h, "poster_tick": poster}  # fmt: skip


# --- other modes ----------------------------------------------------------------------------------------


def still(game: str, level: int, seed: int, out: Path, name: str) -> None:
    """The start frame, and every tile of the level file by sprite name (from the game's LevelMapping)."""
    env = GvgaiEnv(game, levels=(level,), engine_seed=seed)
    frame, _ = env.reset(options={"level": level})
    env.close()
    gdir = ROOT / "engine" / "games" / f"{game}_v0"
    rows = [r for r in (gdir / f"{game}_lvl{level}.txt").read_text().splitlines() if r.strip()]
    mapping, in_map = {}, False
    for line in (gdir / f"{game}.txt").read_text().splitlines():
        s = line.strip()
        if s == "LevelMapping":
            in_map = True
        elif in_map and s and ">" in s:
            ch, sprites = s.split(">", 1)
            mapping[ch.strip()] = sprites.split()
        elif in_map and s and ">" not in s:
            in_map = False
    tile = frame.shape[1] // len(rows[0])
    assert tile * len(rows) == frame.shape[0], "frame isn't a whole number of tiles"
    objects: dict[str, list] = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            for sprite in mapping.get(ch, []):
                if sprite in ("floor", "wall"):
                    continue
                objects.setdefault(sprite, []).append({
                    "col": c, "row": r, "x": c * tile * SCALE, "y": r * tile * SCALE,
                    "w": tile * SCALE, "h": tile * SCALE,
                    "center": [(c + 0.5) * tile * SCALE, (r + 0.5) * tile * SCALE],
                })  # fmt: skip
    # what each object does: the game's interaction and termination rules that name it
    text = (gdir / f"{game}.txt").read_text().split("InteractionSet", 1)[-1].splitlines()
    rules = {o: [ln.strip() for ln in text if o in ln.replace(">", " ").split()] for o in objects}
    Image.fromarray(upscale(frame)).save(out / f"{name}.png", optimize=True)
    info = {"game": game, "level": level, "seed": seed, "grid": [len(rows[0]), len(rows)],
            "tile_px_native": tile, "scale": SCALE, "tile_px": tile * SCALE,
            "image": [frame.shape[1] * SCALE, frame.shape[0] * SCALE],
            "coords": "pixels in the scaled PNG, top-left origin; x, y is the tile's top-left corner",
            "level_file": rows, "objects": objects, "rules": rules}  # fmt: skip
    (out / f"{name}.json").write_text(json.dumps(info, indent=1))
    print(f"{name}.png: {frame.shape[1]}x{frame.shape[0]} native, tile {tile}px; " + ", ".join(objects))


def grid_png(act: torch.Tensor, path: Path, max_ch: int = 16) -> None:
    """Up to 16 channels in a 4-wide grid, each normalised to its own min..max, enlarged to ~48 px high."""
    a = act[:max_ch].float().numpy()
    c, h, w = a.shape
    k = max(1, 48 // h)
    cols = min(4, c)
    rows = -(-c // cols)
    gap = 2
    img = np.full((rows * (h * k + gap) - gap, cols * (w * k + gap) - gap), 40, np.uint8)
    for i in range(c):
        ch = a[i]
        lo, hi = ch.min(), ch.max()
        g = np.zeros_like(ch) if hi - lo < 1e-8 else (ch - lo) / (hi - lo)
        g = (g * 255).round().astype(np.uint8).repeat(k, 0).repeat(k, 1)
        r0, c0 = (i // cols) * (h * k + gap), (i % cols) * (w * k + gap)
        img[r0 : r0 + h * k, c0 : c0 + w * k] = g
    Image.fromarray(img).save(path, optimize=True)


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def features(rec: dict, impala_path: str, impala_preset: str, out: Path, name: str, tick: int | None) -> None:
    """Activations of the 2018 network and the IMPALA network on one moment of the recorded 2018 game.
    Each net gets its own input, as it would at that moment: the 2018 agent's actual stack (4 consecutive ticks,
    Lanczos), and for IMPALA the gray-half frames at that tick and 4, 8 and 12 ticks before."""
    decs = rec["decisions"]
    i = min(range(len(decs)), key=lambda j: abs(decs[j]["tick"] - (tick if tick is not None else rec["ticks"] // 2)))
    t = decs[i]["tick"]
    s2018 = rec["stacks"][i]
    frames = rec["frames"]
    s_imp = np.stack([gray_half(frames[max(0, t - 4 * k)]) for k in (3, 2, 1, 0)])
    Image.fromarray(upscale(frames[t])).save(out / f"{name}-frame.png", optimize=True)
    for label, s in (("2018", s2018), ("impala", s_imp)):
        strip = np.concatenate([np.pad(f, ((0, 0), (0, 2)), constant_values=40) for f in s], 1)
        Image.fromarray(strip.repeat(SCALE, 0).repeat(SCALE, 1)).save(out / f"{name}-{label}-input.png", optimize=True)
    names = rec["names"]
    nets = {
        "2018": load_2018(rec["weights"], *s2018.shape[1:], len(names)),
        "impala": load_modern(impala_path, impala_preset, *s_imp.shape[1:], len(names)),
    }
    desc: dict = {"game": rec["game"], "level": rec["level"], "seed": rec["seed"], "tick": t, "actions": names,
                  "normalisation": "each channel scaled to its own min..max, greyscale; up to 16 channels",
                  "nets": {}}  # fmt: skip
    for label, net in nets.items():
        x = torch.from_numpy((s2018 if label == "2018" else s_imp)[None])
        layers: list[dict] = [{"name": "input", "kind": "frames", "shape": list(x.shape[1:]), "params": 0,
                               "png": f"{name}-{label}-input.png"}]  # fmt: skip
        acts: dict[str, torch.Tensor] = {}
        hooks = []
        if label == "2018":
            watch = {"conv1": (net.conv[1], net.conv[0]), "conv2": (net.conv[3], net.conv[2]),
                     "conv3": (net.conv[5], net.conv[4])}  # fmt: skip
        else:
            watch = {}
            for s in range(3):
                conv, pool, r1, r2 = net.conv[4 * s : 4 * s + 4]
                watch |= {f"stage{s + 1}.conv": (conv, conv), f"stage{s + 1}.pool": (pool, pool),
                          f"stage{s + 1}.res1": (r1, r1), f"stage{s + 1}.res2": (r2, r2)}  # fmt: skip
        for key, (mod, _) in watch.items():
            hooks.append(mod.register_forward_hook(lambda _m, _i, o, key=key, acts=acts: acts.__setitem__(key, o[0])))
        with torch.no_grad():
            q = net(x)[0] if label == "2018" else net.q(x)[0]
        for hk in hooks:
            hk.remove()
        for key, (_, owner) in watch.items():
            a = acts[key]
            png = f"{name}-{label}-{key.replace('.', '-')}.png"
            grid_png(a, out / png)
            kind = type(owner).__name__
            entry = {"name": key, "kind": kind, "shape": list(a.shape), "params": n_params(owner), "png": png}
            if isinstance(owner, torch.nn.Conv2d):
                entry |= {"kernel": owner.kernel_size[0], "stride": owner.stride[0]}
            if label == "2018":
                entry["after"] = "ReLU"
            layers.append(entry)
        heads = [("head.0", net.head[0]), ("head.2", net.head[2])] if label == "2018" else [
            ("adv.0", net.adv[0]), ("adv.2", net.adv[2])]  # fmt: skip
        for key, lin in heads:
            layers.append({"name": key, "kind": "Linear", "shape": [lin.out_features], "in": lin.in_features,
                           "params": n_params(lin)})  # fmt: skip
        desc["nets"][label] = {"layers": layers, "total_params": n_params(net),
                               "q": [round(float(v), 4) for v in q]}  # fmt: skip
    (out / f"{name}.json").write_text(json.dumps(desc, indent=1))
    print(f"features at tick {t}: " + ", ".join(f"{k} {v['total_params']:,} params" for k, v in desc["nets"].items()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--agent",
        help="ddqn2018:<pt> | modern:<pt>:<preset> | hold:<action> | random[:n] | mcts:<iters> | jev[:model[:warn]]"
        " | rule",
    )
    p.add_argument("--game", default="aliens")
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--seed", type=int, default=0, help="engine seed")
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--fps", type=float, default=30)
    p.add_argument("--max-ticks", type=int, default=None, help="cut the video after this many ticks")
    p.add_argument("--webm-crf", type=int, default=36)
    p.add_argument("--no-video", action="store_true", help="play and print the result only")
    p.add_argument("--verify", action="store_true", help="also replay through the evaluation's own loop")
    p.add_argument("--still", action="store_true", help="only the start frame and the level's tile positions")
    p.add_argument("--features", metavar="IMPALA_PT", help="with a ddqn2018 agent: layer activations")
    p.add_argument("--features-preset", default="dqn-impala")
    p.add_argument("--features-tick", type=int, default=None)
    a = p.parse_args()
    torch.set_num_threads(2)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.still:
        still(a.game, a.level, a.seed, out, a.name)
        return
    parts = a.agent.split(":")
    kind = parts[0]
    if kind == "ddqn2018":
        rec = play_2018(parts[1], a.game, a.level, a.seed)
    elif kind == "modern":
        rec = play_modern(parts[1], parts[2], a.game, a.level, a.seed)
    elif kind in ("hold", "random"):
        rec = play_trivial(kind, parts[1] if len(parts) > 1 else "", a.game, a.level, a.seed)
    elif kind == "jev":
        rec = play_jev(parts[1] if len(parts) > 1 else "jev-latest", "warn" in parts[2:], a.game, a.level, a.seed)
    elif kind == "rule":
        rec = play_rule(a.game, a.level, a.seed)
    elif kind == "mcts":
        rec = play_mcts(int(parts[1]), int(parts[2]) if len(parts) > 2 else 0, a.game, a.level, a.seed)
    else:
        raise SystemExit(f"unknown agent {a.agent}")
    print(f"{a.name}: {'win' if rec['win'] else 'loss'}, score {rec['score']:g}, {rec['ticks']} ticks")
    check = verify(kind, parts, a.game, a.level, a.seed, rec) if a.verify else None
    if check:
        print(f"verify: {check}")
    if a.features:
        rec |= {"weights": parts[1], "game": a.game, "level": a.level, "seed": a.seed}
        features(rec, a.features, a.features_preset, out, a.name, a.features_tick)
        return
    if a.no_video:
        return
    video = write_clip(rec, out, a.name, a.fps, a.max_ticks, a.webm_crf)
    meta = {"game": a.game, "level": a.level, "seed": a.seed, "agent": a.agent, "agent_notes": rec["agent_notes"],
            "win": rec["win"], "score": rec["score"], "ticks": rec["ticks"], "action_names": rec["names"],
            "verified": check, "video": video,
            "tick_meaning": "ticks since the game's start; frame i of the video shows the screen after tick i; a "
                            "decision made on tick t plays from tick t + 1",
            "command": "uv run python -m tools.record " + " ".join("<out>" if x == a.out else x for x in sys.argv[1:]),
            "decisions": rec["decisions"]}  # fmt: skip
    (out / f"{a.name}.json").write_text(json.dumps(meta, separators=(",", ":")))
    sizes = ", ".join(f"{ext} {os.path.getsize(out / f'{a.name}.{ext}') // 1024} KB" for ext in ("mp4", "webm", "png"))
    print(f"wrote {out / a.name}.*: {sizes}")


if __name__ == "__main__":
    main()
