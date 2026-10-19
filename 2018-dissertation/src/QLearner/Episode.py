class Episode:
    def __init__(self, episode_number):
        self.episode_number = episode_number
        self.win = False
        self.current_step = 0
        self.total_reward = 0

    def add_reward(self, reward):
        self.total_reward += reward

    def set_win(self, win):
        self.win = win

