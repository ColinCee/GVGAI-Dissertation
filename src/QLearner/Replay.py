import random

from SumTree import SumTree


class Replay:
    def __init__(self, memory_size):
        self.PER_alpha = 0.6
        self.PER_beta = 1e-3
        self.memory = SumTree(memory_size)

    def add_sample(self, priority, data):
        self.memory.add(p=priority, data=data)

    def get_sample(self):
        p_i = random.randint(0, int(self.memory.total()))
        return self.memory.get(p_i)

    def update_sample(self, idx, priority):
        self.memory.update(idx=idx, p=priority)

    def get_priority(self, prediction, target):
        error = abs(prediction - target)
        return (error + self.PER_beta) ** self.PER_alpha

    def update_priorities(self, sample_batch):
        for idx, priority, sample in sample_batch:
            self.update_sample(idx, priority)


class Sample:
    def __init__(self, state, action_string, reward, next_state, done):
        self.state = state
        self.action_string = action_string
        self.reward = reward
        self.next_state = next_state
        self.done = done