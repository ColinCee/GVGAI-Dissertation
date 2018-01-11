import sys


class CompetitionParameters:
    """
     * Competition parameters, should be the same with the ones on Server side:
     * refer to core.competition.CompetitionParameters
    """
    def __init__(self):
        pass

    if 'win32' in sys.platform:
        OS_WIN = True
    else:
        OS_WIN = False

    USE_SOCKETS = True
    MILLIS_IN_MIN = 60*1000
    START_TIME = 1*MILLIS_IN_MIN
    INITIALIZATION_TIME = 1*MILLIS_IN_MIN
    ACTION_TIME = 1*MILLIS_IN_MIN
    ACTION_TIME_DISQ = 1*MILLIS_IN_MIN
    TOTAL_LEARNING_TIME = 480*MILLIS_IN_MIN
    EXTRA_LEARNING_TIME = 2*MILLIS_IN_MIN
    SOCKET_PORT = 8080
    SCREENSHOT_FILENAME = "gameStateByBytes.png"