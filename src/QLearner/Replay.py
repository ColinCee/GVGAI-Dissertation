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
