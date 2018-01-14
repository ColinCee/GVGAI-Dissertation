import cv2
import shutil
from CompetitionParameters import CompetitionParameters
from SerializableStateObservation import SerializableStateObservation
import numpy as np
import os
from collections import deque

class State:

    def __init__(self, max_frames):
        self.max_frames = max_frames
        self.frames = deque(maxlen=self.max_frames)

    def get_frame_stack(self):
        assert len(self.frames) == self.max_frames, "{} - {}".format(len(self.frames), self.max_frames)
        stacks = np.stack(self.frames, axis=2)
        return stacks

    def add_frame(self):
        self.frames.append(self.get_single_frame(CompetitionParameters.SCREENSHOT_FILENAME))

    @staticmethod
    def get_single_frame(image_name):
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        resized_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        return np.array(resized_image)

    def save_game_state(self, nb_episode, episode_step):
        # Save a snapshot of the game, both original and resized

        resized_image = self.frames[-1]
        srcfile = CompetitionParameters.SCREENSHOT_FILENAME
        dstfolder = "game-snapshots/Episode {}/".format(nb_episode)

        # Set the file names
        original_file = os.path.join(dstfolder, "original/Step - {}.png".format(episode_step))
        resized_file = os.path.join(dstfolder, "resized/Step - {}.png".format(episode_step))

        # Create the folders if they don't exist
        if not os.path.exists(os.path.dirname(original_file)):
            os.makedirs(os.path.dirname(original_file))
        if not os.path.exists(os.path.dirname(resized_file)):
            os.makedirs(os.path.dirname(resized_file))

        # Copy!
        shutil.copy(srcfile, original_file)
        cv2.imwrite(resized_file, resized_image)
