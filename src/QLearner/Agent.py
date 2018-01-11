import datetime
import random

import cv2
import shutil
from AbstractPlayer import AbstractPlayer
from CompetitionParameters import CompetitionParameters
from SerializableStateObservation import SerializableStateObservation
from PIL import Image
from Types import *
from Brain import Brain
import os
import numpy as np
import matplotlib.pyplot as plt
import weights


class Agent(AbstractPlayer):

    def __init__(self):
        AbstractPlayer.__init__(self)
        self.lastSsoType = LEARNING_SSO_TYPE.BOTH
        self.brain = None
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = 0
        self.prev_game_score = 0
        self.episode_step = 0
        self.episode_reward_history = []

        self.episodes = 0
        self.plot_frequency = 5  # how often to plot filters, in episodes
        self.snapshot_frequency = 50  # how often to save a picture of the game, in steps

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
            image = self.get_state()
            print("Image size is: " + str(image.shape))
            self.brain = Brain(image.shape, sso.availableActions)

        current_state = self.get_state()
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
        self.episodes += 1
        # calculate the terminal reward, the previous reward assumed the game was still in progress
        self.prev_reward = self.calculate_reward(sso)
        self.episode_reward_history.append(self.prev_reward)
        # We need one final remember when we get into the terminal state
        self.brain.remember(self.prev_state, self.prev_action, self.prev_reward, self.get_state(), 1)
        self.brain.replay()
        self.brain.save_model(self.episodes)

        if self.episodes % self.plot_frequency == 0:
            weights.plot_all_layers(self.brain.model, self.episodes)

        # culmulative_reward = [0]
        # for r in self.episode_reward_history:
        #     culmulative_reward.append(r + culmulative_reward[-1])
        #
        # plt.figure()
        # plt.plot(culmulative_reward)
        # plt.show()

        win = True if "WIN" in sso.gameWinner else False
        avg_reward = sum(self.episode_reward_history) / float(len(self.episode_reward_history))
        print("{}. Win: {} | Avg. Reward: {:.5f} | Game Ticks: {} | Epsilon: {:.3f}".format(self.episodes,
                                                                                            win,
                                                                                            avg_reward,
                                                                                            sso.gameTick,
                                                                                            self.brain.exploration_rate))
        # cumulative_reward = [0]
        # for r in self.reward_history:
        #     cumulative_reward.append(r + cumulative_reward[-1])
        # plt.plot(cumulative_reward)
        # plt.show()

        return random.randint(0, 2)

    def get_state(self):
        image = cv2.imread(CompetitionParameters.SCREENSHOT_FILENAME, cv2.IMREAD_COLOR)

        resized_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        self.save_game_state(resized_image)

        return np.array(resized_image)

    def calculate_reward(self, sso: 'SerializableStateObservation'):

        if sso.isGameOver:

            if "WIN" in sso.gameWinner:
                reward = 1
            else:
                reward = -1
        else:
            score_diff = sso.gameScore - self.prev_game_score
            reward = score_diff / 200

        return reward

    def save_game_state(self, resized_image):

        if self.episode_step == 0:
            return
        # Save a snapshot of the game, both original and resized
        if self.episode_step % self.snapshot_frequency == 0:
            srcfile = CompetitionParameters.SCREENSHOT_FILENAME
            dstfolder = "game-snapshots/Episode {}/".format(self.episodes + 1)

            # Set the file names
            original_file = os.path.join(dstfolder, "original/Step - {}.png".format(self.episode_step))
            resized_file = os.path.join(dstfolder, "resized/Step - {}.png".format(self.episode_step))

            # Create the folders if they don't exist
            if not os.path.exists(os.path.dirname(original_file)):
                os.makedirs(os.path.dirname(original_file))
            if not os.path.exists(os.path.dirname(resized_file)):
                os.makedirs(os.path.dirname(resized_file))

            # Copy!
            shutil.copy(srcfile, original_file)
            cv2.imwrite(resized_file, resized_image)
