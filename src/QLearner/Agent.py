import random

from AbstractPlayer import AbstractPlayer
from SerializableStateObservation import SerializableStateObservation
from Types import *

import weights
from Brain import Brain
from State import State
from Statistics import Statistics


class Agent(AbstractPlayer):

    def __init__(self):
        AbstractPlayer.__init__(self)
        self.lastSsoType = LEARNING_SSO_TYPE.BOTH
        self.brain = None
        self.warmup_steps = 1e4
        self.steps_between_training = 5e3
        self.img_stacks = 3

        self.prev_state = None
        self.prev_action = None
        self.prev_reward = 0
        self.prev_game_score = 0
        self.snapshot_frequency = 10
        self.state = State(self.img_stacks)
        self.statistics = Statistics()

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
            # self.brain.load_model()
        current_step = self.statistics.get_episode_step()
        if current_step % self.img_stacks != 0 or current_step == 0:
            self.state.add_frame()
            self.statistics.increment_episode_step()
            # Repeat previous move during frame skip if possible
            if self.prev_action is None:
                return ACTIONS.ACTION_NIL
            else:
                return self.prev_action
        else:
            current_state = self.state.get_frame_stack()
            action = self.brain.get_action(current_state, sso.availableActions)
            if self.prev_state is not None:
                self.brain.remember(self.prev_state, self.prev_action, self.prev_reward, current_state, 0)

            self.prev_state = current_state
            self.prev_action = action
            self.prev_reward = self.calculate_reward(sso)
            self.statistics.add_reward(self.prev_reward)
            self.prev_game_score = sso.gameScore
            self.statistics.steps_since_last_train += 1

        self.save_game_state()
        self.statistics.increment_episode_step()
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
        # Add the new frame
        self.state.add_frame()
        # Remove last reward since we assumed the game was not over
        self.statistics.add_reward(-self.prev_reward)
        # calculate the terminal reward, the previous reward assumed the game was still in progress
        self.prev_reward = self.calculate_reward(sso)
        self.statistics.add_reward(self.prev_reward)
        # We need one final remember when we get into the terminal state
        self.brain.remember(self.prev_state, self.prev_action, self.prev_reward, self.state.get_frame_stack(), 1)
        # Train after we have filled our replay memory a little, also delay training
        episode_count = self.statistics.get_episode_count()
        steps_since_last_trained = self.statistics.steps_since_last_train
        self.statistics.output_episode_stats(sso, self.brain.exploration_rate)
        if len(self.brain.memory) >= self.warmup_steps and steps_since_last_trained >= self.steps_between_training:
            self.brain.replay()
            self.brain.save_model(episode_count)
            weights.plot_all_layers(self.brain.model, episode_count)
            self.statistics.output_training_stats()

        self.statistics.start_new_episode()
        self.save_game_state()
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

    # Save snapshots of a full game
    def save_game_state(self):
        if self.statistics.get_episode_count() % self.snapshot_frequency == 0:
            self.state.save_game_state(self.statistics.get_episode_count(), self.statistics.get_episode_step())