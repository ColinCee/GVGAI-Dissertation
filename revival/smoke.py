"""Phase 0 check: play random games, print frame/action facts and env speed."""
import argparse
import random
import time

from PIL import Image

from gvgai.env import GvgaiEnv

p = argparse.ArgumentParser()
p.add_argument("--game", default="aliens")
p.add_argument("--episodes", type=int, default=5)
p.add_argument("--save-frame", default="")
p.add_argument("--repeat", type=int, default=1, help="ticks per chosen action (2018 used 4)")
a = p.parse_args()

t0 = time.perf_counter()
env = GvgaiEnv(a.game)
print(f"startup {time.perf_counter() - t0:.1f}s")
print("actions", env.action_names)
print("obs shape", env.observation_space.shape)

ticks = 0
t0 = time.perf_counter()
for ep in range(a.episodes):
    obs, info = env.reset()
    total = 0.0
    t = 0
    while True:
        if t % a.repeat == 0:
            act = random.randrange(env.action_space.n)
        t += 1
        obs, r, term, trunc, info = env.step(act)
        total += r
        ticks += 1
        if term or trunc:
            break
    print(f"ep {ep} level {info['level']} ticks {info['tick']} score {info['score']} "
          f"reward {total} winner {info['winner']} timeout {info['timeout']}")
    if a.save_frame and ep == 0:
        Image.fromarray(obs).save(a.save_frame)
dt = time.perf_counter() - t0
print(f"{ticks} ticks in {dt:.1f}s = {ticks / dt:.0f} ticks/s")
grey = Image.fromarray(obs).convert("L")
print("downscaled (2018 style)", grey.resize((grey.width // 2, grey.height // 2)).size[::-1])
env.close()
