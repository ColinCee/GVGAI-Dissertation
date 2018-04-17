import random

from AbstractPlayer import AbstractPlayer
from SerializableStateObservation import SerializableStateObservation
from Types import *

from Brain import Brain
from State import State
from Statistics import Statistics


class Agent(AbstractPlayer):

    def __init__(self):
        AbstractPlayer.__init__(self)
        self.lastSsoType = LEARNING_SSO_TYPE.IMAGE
        self.brain = None
        self.frames_per_stack = 4
        self.frame_downscaling_factor = 2
        self.warmup_stacks = 2e4
        self.target_update_frequency = 5e3  # Update frequency in stacks
        self.prev_state = None
        self.prev_action = None
        self.prev_reward = 0
        self.prev_game_score = 0
        self.snapshot_frequency = 100
        self.validation = True
        self.warmup_phase = True

        self.state = State(self.frames_per_stack, self.frame_downscaling_factor)
        self.statistics = Statistics(self.validation)
        self.transitions = []

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
        self.statistics.start_new_episode()
        self.state.start_new_episode()
        self.transitions = []
        self.warmup_phase = self.statistics.total_stacks < self.warmup_stacks

        # Load from a previous save?
        # self.brain.load_model(self.brain.weight_backup)
        # self.brain.exploration_rate = 0
        # self.validation = True
        # self.statistics.total_steps = int(198176 - self.warmup_steps)
        # self.statistics.episide_count = 1400
        # self.statistics.get_current_episode().episode_number = 1400

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
            self.init_brain(sso)
        # Ignore the first 2 frames (Redundant)
        if sso.gameTick <= 1:
            return ACTIONS.ACTION_NIL

        self.state.add_frame(sso.imageArray)
        current_step = self.statistics.get_current_episode_step()

        # Check if a new stack is available to be processed, also skip initial step
        if current_step % self.frames_per_stack == 0 and current_step > 0:
            current_state = self.state.get_frame_stack()
            action = self.brain.get_action(current_state, sso.availableActions)
            self.prev_state = current_state
            self.prev_action = action
            self.prev_reward = self.calculate_reward(sso)
            self.statistics.add_reward_to_current_episode(self.prev_reward)
        else:
            # Repeat previous move during frame skip if possible
            if self.prev_action is None:
                action = ACTIONS.ACTION_NIL
            else:
                action = self.prev_action

        self.statistics.increment_current_episode_step()
        self.save_game_state()
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
        self.state.add_final_frame()
        self.save_game_state()
        # self.save_model_weights()

        # calculate the terminal reward, the previous reward assumed the game was still in progress
        self.prev_reward = self.calculate_reward(sso)
        self.statistics.add_reward_to_current_episode(self.prev_reward)
        self.statistics.finish_episode(sso, self.brain.exploration_rate)
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
            self.prev_game_score = sso.gameScore
            if score_diff > 0:
                return 1
            elif score_diff < 0:
                return -1
            else:
                return 0

    # Save snapshots of a full game
    def save_game_state(self):
        if (self.statistics.get_episode_count() % self.snapshot_frequency == 0
            and self.statistics.total_steps >= self.warmup_stacks) \
                or self.validation:
            self.state.save_game_state(self.statistics.get_episode_count(), self.statistics.get_current_episode_step())

    def init_brain(self, sso):
        x, y, z = self.state.get_image_dimensions(sso.imageArray)
        new_x = int(x / self.frame_downscaling_factor)
        new_y = int(y / self.frame_downscaling_factor)
        input_shape = (new_x, new_y, self.frames_per_stack)
        print(input_shape)
        self.brain = Brain(input_shape, sso.availableActions)

        if self.validation:
            self.brain.load_model("QLearner/model-backup/weight_backup.h5")
            self.brain.exploration_rate = 0
