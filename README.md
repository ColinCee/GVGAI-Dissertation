# GVGAI-Dissertation

Deep Q-learning agents playing four GVGAI games (Aliens, Chase, Deceptizelda, Wafer Thin Mints), compared with MCTS and a human: Colin Cheung's 2018 undergraduate dissertation, and its 2026 revival.

| Folder | What | State |
|---|---|---|
| [2018-dissertation/](2018-dissertation/) | The 2018 code as it was on `master` (an older plain-DQN version) | Frozen; Python 3.6, Keras 2, TensorFlow 1; no longer installs |
| [2026-revival/](2026-revival/) | The rebuild: the 2018 agent replayed exactly, today's methods, an engine ~75× faster per game | Runs on Windows with Python 3.14, PyTorch 2.14, JDK 25 |

The 2018 variants each lived on their own branch; they are frozen under tags `2018-dqn`, `2018-ddqn`, `2018-ddqn-per` (the final agent) and `2018-dueling`. The report itself is in a separate repository (Dissertation-Report); the 2018 training data it was built from is copied into [2026-revival/results/2018/](2026-revival/results/2018/).

The state at the end of the 2026 round (19 October 2026) is tagged `revival-2026`.

**Start here:** findings in [2026-revival/RESULTS.md](2026-revival/RESULTS.md); coming back after years, [2026-revival/LEARNINGS.md](2026-revival/LEARNINGS.md).
