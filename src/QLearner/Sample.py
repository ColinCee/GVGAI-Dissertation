class Sample():
    def __init__(self, state, action_string, reward, next_state, done):
        self.state = state
        self.action_string = action_string
        self.reward = reward
        self.next_state = next_state
        self.done = done
