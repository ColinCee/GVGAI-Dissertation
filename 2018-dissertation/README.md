# The 2018 code

DQN for GVGAI as it was on `master` in 2018: an older plain-DQN version, not the final agent. The final agent (DDQN + prioritised replay + the loss-penalty reward) is at tag `2018-ddqn-per`; the other variants at `2018-dqn`, `2018-ddqn` and `2018-dueling`. Each tag checks out the whole 2018 repository as it was.

How it ran in 2018: clone the GVGAI repository, paste this `src/` into its Python client folder, and run from there with Anaconda 3 and keras, tensorflow, h5py, matplotlib, pillow and opencv. Those versions (Python 3.6, Keras 2, TensorFlow 1) no longer install; [../2026-revival/](../2026-revival/) has a faithful PyTorch port (`agents/ddqn2018.py`) that reproduces the 2018 results. The 2018 training logs are in [../2026-revival/results/2018/](../2026-revival/results/2018/).
