from Episode import Episode


class Statistics():
    def __init__(self):
        self.episode_history = []
        self.total_steps = 0
        self.total_stacks = 0
        self.train_count = 0
        self.update_count = 0
        self.stacks_since_last_update = 0

    def get_current_episode(self) -> 'Episode':
        return self.episode_history[-1]

    # Adds the reward to the current episode history
    def add_reward_to_current_episode(self, reward):
        self.get_current_episode().add_reward(reward)

    def get_current_episode_step(self):
        return self.get_current_episode().current_step

    def get_episode_count(self):
        return len(self.episode_history)

    def increment_current_episode_step(self):
        self.get_current_episode().current_step += 1

    def start_new_episode(self):
        self.episode_history.append(Episode(self.get_episode_count() + 1))

    def finish_episode(self, sso, exploration_rate):
        self.increment_current_episode_step()
        self.stacks_since_last_update += 1
        self.total_steps += self.get_current_episode_step()
        self.output_episode_stats(sso, exploration_rate)

    def log_train(self):
        self.train_count += 1

    def log_update(self):
        self.update_count += 1
        self.stacks_since_last_update = 0

    def output_episode_stats(self, sso, exploration_rate):
        current_episode = self.get_current_episode()
        win = "WIN" in sso.gameWinner
        current_episode.set_win(win)

        print(
            "{}. Win: {} | "
            "Tot. Reward: {:.3f} | "
            "Game Ticks: {:3d} | "
            "Epsilon: {:.3f} | "
            "Total Stacks: {:<6d} | "
            "Total Trains: {}".format(
                current_episode.episode_number,
                current_episode.win,
                current_episode.total_reward,
                sso.gameTick,
                exploration_rate,
                self.total_stacks,
                self.train_count))

        self.log_episode_stats(exploration_rate, sso.gameScore)

    def log_episode_stats(self, exploration_rate, game_score):
        with open('reward_history.csv', 'a+') as file:
            win_as_int = 1 if self.get_current_episode().win else 0
            file.write(
                "{}, {:.3f}, {}, {}, {:.3f}\n".format(game_score,
                                                      self.get_current_episode().total_reward,
                                                      self.get_current_episode_step(),
                                                      win_as_int, exploration_rate))
        file.close()
