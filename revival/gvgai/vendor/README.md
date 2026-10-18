Python client for the GVGAI Java learning server, copied from
[GVGAI_GYM](https://github.com/rubenrtorrado/GVGAI_GYM) at commit
`15e0a86dcaa138721409b07ed970255485e150d6` (`gym_gvgai/envs/gvgai/clients/GVGAI-PythonClient/src/utils`), Apache 2.0.

Changes from upstream, all in `ClientCommGYM.py` unless named:
- Pillow instead of `scipy.misc.imread` (removed in SciPy 1.2); `is ''` → `== ''`.
- Each tick's frame is decoded in memory instead of written to disk and read back. The end-of-game frame still comes from the file the server writes.
- Runs the real JVM, not the Windows `javapath` stub (killing the stub leaves the JVM running), with extra JVM options from `JAVA_OPTS`.
- The server's stdin/stderr go to DEVNULL too, so it cannot hold the caller's pipes open.
- `IOSocket.py`: connection prints silenced.
