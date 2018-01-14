import csv


class Statistics():
    def __init__(self):
        self.episide_count = 0
        self.episodes_since_last_train = 0
        self.steps_since_last_train = 0
        self.episode_history = []  # Tracks the total reward at the end of each game

    # Adds the reward to the current episode history
    def add_reward(self, reward):
        self.get_current_episode().add_reward(reward)

    def get_episode_step(self):
        return self.get_current_episode().step

    def increment_episode_step(self):
        self.get_current_episode().step += 1

    def get_episode_count(self):
        return self.episide_count

    def start_new_episode(self):
        self.episide_count += 1
        self.episode_history.append(Episode(self.get_episode_count()))

    def get_current_episode(self) -> 'Episode':
        return self.episode_history[-1]

    def reset_on_train(self):
        self.steps_since_last_train = 0
        self.episodes_since_last_train = 0
        self.episode_history = []

    def output_episode_stats(self, sso, exploration_rate):
        self.steps_since_last_train += 1
        self.episodes_since_last_train += 1
        self.get_current_episode().output_episode_stats(sso, exploration_rate)

    def output_training_stats(self):

        mean_episode_reward = 0
        for episode in self.episode_history[-self.episodes_since_last_train:]:
            mean_episode_reward += episode.total_reward
        mean_episode_reward /= self.episodes_since_last_train

        print("[{}]. Eps since last train - with mean reward: {:2.2f}".format(self.episodes_since_last_train,
                                                                              mean_episode_reward))
        self.log_training_to_csv()

    def log_training_to_csv(self):
        with open('reward_history.csv', 'a+') as file:
            for episode in self.episode_history[-self.episodes_since_last_train:]:  # type: Episode
                file.write("{},\n".format(episode.total_reward))
        file.close()


class Episode:
    def __init__(self, episode_number):
        self.step = 0
        self.episode_number = episode_number
        self.total_reward = 0

    def add_reward(self, reward):
        self.total_reward += reward

    def output_episode_stats(self, sso, exploration_rate):
        win = "True " if "WIN" in sso.gameWinner else "False"
        print(
            "{}. Win: {} | "
            "Tot. Reward: {:2d} | "
            "Game Ticks: {:3d} | "
            "Epsilon: {:.3f} | ".format(
                self.episode_number,
                win,
                self.total_reward,
                sso.gameTick,
                exploration_rate))
