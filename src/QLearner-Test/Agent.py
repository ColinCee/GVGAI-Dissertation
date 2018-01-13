import random

import cv2
from AbstractPlayer import AbstractPlayer
from CompetitionParameters import CompetitionParameters
from Types import LEARNING_SSO_TYPE

from Brain import Brain
import numpy as np


class Agent(AbstractPlayer):
    def __init__(self):
        AbstractPlayer.__init__(self)
        self.lastSsoType = LEARNING_SSO_TYPE.BOTH
        self.episodes = 0
        self.brain = None


    def init(self, sso, elapsedTimer):
        pass

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

    def act(self, sso: 'SerializableStateObservation', elapsedTimer):
        # Start the model
        if self.brain is None:
            self.brain = Brain(self.get_state().shape, sso.availableActions)
            self.brain.load_model()
        # Simple forward pass to get action
        action = self.brain.get_action(self.get_state(), sso.availableActions)

        return action

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

    def result(self, sso, elapsedTimer):
        self.episodes += 1
        win = "True " if "WIN" in sso.gameWinner else "False"
        print("{}. Win: {} | Game Ticks: {} | Epsilon: {:.3f}".format(self.episodes,
                                                                      win,
                                                                      sso.gameTick,
                                                                      self.brain.exploration_rate))
        return random.randint(0, 2)

    def get_state(self):
        image = cv2.imread(CompetitionParameters.SCREENSHOT_FILENAME, cv2.IMREAD_COLOR)

        resized_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        return np.array(resized_image)
