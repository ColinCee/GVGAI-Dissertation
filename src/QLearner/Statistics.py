class Statistics():
    def __init__(self):
        self.total_steps = 0
        self.episide_count = 0
        self.train_count = 0
        self.steps_since_last_train = 0
        self.steps_since_last_update = 0
        self.current_episode = Episode(0)  # Tracks the total reward at the end of each game
        self.mean_reward_since_train = 0
        self.mean_duration_since_train = 0

    # Adds the reward to the current episode history
    def add_reward(self, reward):
        self.get_current_episode().add_reward(reward)

    def get_episode_step(self):
        return self.get_current_episode().current_step

    def increment_episode_step(self):
        self.get_current_episode().current_step += 1

    def increment_train_update_steps(self):
        self.steps_since_last_train += 1
        self.steps_since_last_update += 1

    def start_new_episode(self):
        self.episide_count += 1
        self.current_episode = Episode(self.episide_count)

    def get_current_episode(self) -> 'Episode':
        return self.current_episode

    def reset_on_train(self):
        self.train_count += 1
        self.steps_since_last_train = 0

    def reset_on_update(self):
        self.steps_since_last_update = 0

    def output_episode_stats(self, sso, exploration_rate, total_steps):
        self.get_current_episode().win = ("WIN" in sso.gameWinner)
        self.get_current_episode().output_episode_stats(sso, exploration_rate, total_steps)
        self.log_episode_stats()

    def log_episode_stats(self):
        with open('reward_history.csv', 'a+') as file:
            file.write("{}, {}\n".format(self.current_episode.total_reward, self.current_episode.current_step,
                                         self.get_current_episode().win))
        file.close()


class Episode:
    def __init__(self, episode_number):
        self.win = False
        self.current_step = 0
        self.episode_number = episode_number
        self.total_reward = 0

    def add_reward(self, reward):
        self.total_reward += reward

    def output_episode_stats(self, sso, exploration_rate, total_steps):
        win = "True " if "WIN" in sso.gameWinner else "False"
        print(
            "{}. Win: {} | "
            "Tot. Reward: {:2d} | "
            "Game Ticks: {:3d} | "
            "Epsilon: {:.3f} | "
            "Total Steps: {:<6d} ".format(
                self.episode_number,
                win,
                self.total_reward,
                sso.gameTick,
                exploration_rate,
                total_steps))
