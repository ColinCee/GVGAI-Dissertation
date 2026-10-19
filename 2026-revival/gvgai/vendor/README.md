Python client for the GVGAI Java learning server, copied from
[GVGAI_GYM](https://github.com/rubenrtorrado/GVGAI_GYM) at commit
`15e0a86dcaa138721409b07ed970255485e150d6` (`gym_gvgai/envs/gvgai/clients/GVGAI-PythonClient/src/utils`), Apache 2.0.

Changes from upstream, all in `ClientCommGYM.py` unless named:
- Pillow instead of `scipy.misc.imread` (removed in SciPy 1.2); `is ''` → `== ''`.
- Frames arrive as raw bytes after each JSON line (`imageBytes`; greyscale and halved when the server runs with `gvgai.grayHalf`), or as an in-memory PNG; the file the server writes is only a fallback when neither came.
- `step_send` / `step_recv` split one step so a caller can send moves to many servers before reading any reply; `act(action, repeat)` sends `ACTION#IMAGE#k` so the server repeats the action k ticks.
- Runs the engine compiled by `engine/build.py` (`CLASSES`).
- Runs the real JVM, not the Windows `javapath` stub (killing the stub leaves the JVM running), with extra JVM options from `JAVA_OPTS`.
- The server's stdin/stderr go to DEVNULL too, so it cannot hold the caller's pipes open.
- `IOSocket.py`: connection prints silenced; buffered reads (`recv_end`, `read_exact` for the frame bytes) that raise `ConnectionError` when the server closes the socket, instead of spinning.
