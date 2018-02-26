import random

from SumTree import SumTree


class Replay:
    def __init__(self, memory_size):
        self.memory = SumTree(memory_size)

    def add_sample(self, data):
        self.memory.add(p=1, data=data)

    def get_sample(self):
        p_i = random.randint(0, int(self.memory.total()))
        return self.memory.get(p_i)

    def update_sample(self, idx, priority):
        self.memory.update(idx=idx, p=priority)


class Sample:
    def __init__(self, state, action_string, reward, next_state, done):
        self.state = state
        self.action_string = action_string
        self.reward = reward
        self.next_state = next_state
        self.done = done
