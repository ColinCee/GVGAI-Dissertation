import random
from collections import deque

class Replay:
    def __init__(self, memory_size):
        self.memory = deque(maxlen=memory_size)

    def add_sample(self, state, action, reward, next_state, done):
        data = Sample(state, action, reward, next_state, done)
        self.memory.append(data)

    def get_sample(self):
        return random.sample(self.memory, 1)[0]


class Sample:
    def __init__(self, state, action_string, reward, next_state, done):
        self.state = state
        self.action_string = action_string
        self.reward = reward
        self.next_state = next_state
        self.done = done
