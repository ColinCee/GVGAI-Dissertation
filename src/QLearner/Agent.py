import random

from AbstractPlayer import AbstractPlayer
from SerializableStateObservation import SerializableStateObservation
from Types import *

import weights
from Brain import Brain
from State import State


class Agent(AbstractPlayer):

    def __init__(self):
        AbstractPlayer.__init__(self)
        self.lastSsoType = LEARNING_SSO_TYPE.BOTH
        self.brain = None
        self.total_episodes = 0
        self.total_steps = 0
        self.warmup_steps = 1e5
        self.steps_between_training = 1e3
        self.snapshot_frequency = 20  # how often to save a full game in episodes
        self.img_stacks = 3

        self.prev_state = None
        self.prev_action = None
        self.prev_reward = 0
        self.prev_game_score = 0
        self.episode_step = 0
        self.episode_reward_history = []
        self.state = State(self.img_stacks)

    def init(self, sso, timer):
        """
        * Public method to be called at the start of every level of a game.
        * Perform any level-entry initialization here.
        * @param sso Phase Observation of the current game.
        * @param elapsedTimer Timer (1s)
        """
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = 0
        self.prev_game_score = 0
        self.episode_step = 0
        self.episode_reward_history = []

    def act(self, sso: 'SerializableStateObservation', timer):
        """
        * Method used to determine the next move to be performed by the agent.
        * This method can be used to identify the current state of the game and all
        * relevant details, then to choose the desired course of action.
        *
        * @param sso Observation of the current state of the game to be used in deciding
        *            the next action to be taken by the agent.
        * @param elapsedTimer Timer (40ms)
        * @return The action to be performed by the agent.
        """
        if self.brain is None:
            self.brain = Brain(sso.availableActions)
            # Load from a previous save?
            #self.brain.load_model()

        if self.episode_step % self.img_stacks != 0 or self.episode_step == 0:
            self.state.add_frame()
            self.episode_step += 1
            # Don't move between frames
            return ACTIONS.ACTION_NIL
        else:
            if self.total_episodes % self.snapshot_frequency == 0:
                self.state.save_game_state(self.total_episodes, self.episode_step)

            current_state = self.state.get_frame_stack()
            action = self.brain.get_action(current_state, sso.availableActions)
            if self.prev_state is not None:
                self.brain.remember(self.prev_state, self.prev_action, self.prev_reward, current_state, 0)

            self.prev_state = current_state
            self.prev_action = action
            self.prev_reward = self.calculate_reward(sso)
            self.episode_reward_history.append(self.prev_reward)
            self.prev_game_score = sso.gameScore
            self.episode_step += 1
            return action

    def result(self, sso: 'SerializableStateObservation', timer):
        """
        * Method used to perform actions in case of a game end.
        * This is the last thing called when a level is played (the game is already in a terminal state).
        * Use this for actions such as teardown or process data.
        *
        * @param sso The current state observation of the game.
        * @param elapsedTimer Timer (up to CompetitionParameters.TOTAL_LEARNING_TIME
        * or CompetitionParameters.EXTRA_LEARNING_TIME if current global time is beyond TOTAL_LEARNING_TIME)
        * @return The next level of the current game to be played.
        * The level is bound in the range of [0,2]. If the input is any different, then the level
        * chosen will be ignored, and the game will play a random one instead.
        """
        self.episode_reward_history.append(self.prev_reward)
        # Add the new frame
        self.state.add_frame()
        self.state.save_game_state(self.total_episodes, self.episode_step)
        # calculate the terminal reward, the previous reward assumed the game was still in progress
        self.prev_reward = self.calculate_reward(sso)

        # We need one final remember when we get into the terminal state
        self.brain.remember(self.prev_state, self.prev_action, self.prev_reward, self.state.get_frame_stack(), 1)

        # Train after we have filled our replay memory a little, also delay training
        if self.total_steps >= self.warmup_steps and self.total_steps % self.steps_between_training == 0:
            mean_episode_reward = sum(self.episode_reward_history[-20:]) / 20
            print("Begin training. Mean reward for past 20 games: {}".format(mean_episode_reward))
            self.brain.replay()
            self.brain.save_model(self.total_episodes)
            weights.plot_all_layers(self.brain.model, self.total_episodes)


        win = "True " if "WIN" in sso.gameWinner else "False"
        avg_reward = sum(self.episode_reward_history) / len(self.episode_reward_history)

        print(
            "{}. Win: {} | Tot. Reward: {:.5f} | Mean Reward {:.5f} | Game Ticks: {} | Epsilon: {:.3f}".format(
                self.total_episodes,
                win,
                sum(self.episode_reward_history),
                avg_reward,
                sso.gameTick,
                self.brain.exploration_rate))

        self.total_episodes += 1
        return random.randint(0, 2)

    # Clip rewards, all positive rewards are set to 1, all negative rewards are set to -1, 0 is unchanged
    def calculate_reward(self, sso: 'SerializableStateObservation'):
        if sso.isGameOver:
            if "WIN" in sso.gameWinner:
                return 1
            else:
                return -1
        else:
            score_diff = (sso.gameScore - self.prev_game_score)

            if score_diff > 0:
                return 1
            else:
                return 0
