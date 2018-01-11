class Episode():
    def __init__(self):
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = 0
        self.prev_game_score = 0
        self.episode_reward_history = []

    def step(self, current_state, action, reward, next_state):
        self.prev_state = current_state
        self.prev_action = action
        self.prev_reward = self.calculate_reward(sso)

