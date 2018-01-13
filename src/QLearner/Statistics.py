import csv

class Statistics():
    def __init__(self):
        self.episode_count = 0
        self.episodes_since_last_train = 0
        self.steps_since_last_train = 0

        self.episode_step = 0
        self.current_episode_rewards = []  # Each game stores the current reward
        self.episode_rewards_history = []  # Tracks the total reward at the end of each game

    # Adds the reward to the current episode history
    def add_reward(self, reward):
        self.current_episode_rewards.append(reward)

    # Used when the game is over and the reward is re-calculated
    def remove_last_reward(self):
        del self.current_episode_rewards[-1]

    def start_new_episode(self):
        self.episode_count += 1
        self.episode_rewards_history.append(sum(self.current_episode_rewards))
        self.reset_on_new_episode()

    def reset_on_new_episode(self):
        self.episode_step = 0
        self.current_episode_rewards = []

    def reset_on_train(self):
        self.episodes_since_last_train = 0
        self.steps_since_last_train = 0

    def output_episode_stats(self, sso, exploration_rate):

        win = "True " if "WIN" in sso.gameWinner else "False"
        print(
            "{}. Win: {} | "
            "Tot. Reward: {:2d} | "
            "Game Ticks: {:3d} | "
            "Epsilon: {:.3f} | ".format(
                self.episode_count,
                win,
                sum(self.current_episode_rewards),
                sso.gameTick,
                exploration_rate))

        self.steps_since_last_train += 1
        self.episodes_since_last_train += 1

    def output_training_stats(self):
        mean_episode_reward = sum(
            self.episode_rewards_history[-self.episodes_since_last_train:]) / self.episodes_since_last_train
        print("[{}]. Eps since last train - with mean reward: {:2.2f}".format(self.episodes_since_last_train,
                                                                              mean_episode_reward))
        self.log_training_to_csv()
        self.reset_on_train()

    def log_training_to_csv(self):
        with open('reward_history.csv', 'a+') as file:
            for reward in self.episode_rewards_history[-self.episodes_since_last_train:]:
                file.write("{},\n".format(reward))
        file.close()