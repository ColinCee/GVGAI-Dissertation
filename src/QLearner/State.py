import io
import os
from collections import deque

import numpy as np
from CompetitionParameters import CompetitionParameters
from PIL import Image


class State:

    def __init__(self, max_frames):
        self.max_frames = max_frames
        self.original_frames = deque(maxlen=self.max_frames)
        self.resized_frames = deque(maxlen=self.max_frames)

    def start_new_episode(self):
        self.original_frames.clear()
        self.resized_frames.clear()

    def get_frame_stack(self):
        assert len(self.resized_frames) == self.max_frames, "{} - {}".format(len(self.resized_frames), self.max_frames)
        stacks = np.stack(self.resized_frames, axis=2)
        return stacks

    def add_frame(self, image_array):
        image = self.get_image_from_array(image_array)
        resized_image = self.resize_image(image)

        self.original_frames.append(np.array(image))
        self.resized_frames.append(np.array(resized_image))

    def add_final_frame(self):
        image = Image.open(CompetitionParameters.SCREENSHOT_FILENAME)
        resized_image = self.resize_image(image)

        self.original_frames.append(np.array(image))
        self.resized_frames.append(np.array(resized_image))

        image.close()  # Close connection to the image
        resized_image.close()

    def get_image_from_array(self, pixels):
        for i, e in enumerate(pixels):
            pixels[i] = e & 0xFF
        image = Image.open(io.BytesIO(bytearray(pixels)))
        return image

    def resize_image(self, image):
        b_and_w = image.convert('L')
        width, height = image.size
        new_width = int(width / 2)
        new_height = int(height / 2)
        resized_image = b_and_w.resize((new_width, new_height), resample=Image.LANCZOS)
        return resized_image

    def save_game_state(self, nb_episode, episode_step):
        # Save a snapshot of the game, both original and resized
        if len(self.original_frames) == 0 or len(self.resized_frames) == 0:
            return

        original_image = self.original_frames[-1]
        resized_image = self.resized_frames[-1]
        dstfolder = "game-snapshots/Episode {}/".format(nb_episode)

        # Set the file names
        original_file = os.path.join(dstfolder, "original/Step - {}.png".format(episode_step))
        resized_file = os.path.join(dstfolder, "resized/Step - {}.png".format(episode_step))

        # Create the folders if they don't exist
        if not os.path.exists(os.path.dirname(original_file)):
            os.makedirs(os.path.dirname(original_file))
        if not os.path.exists(os.path.dirname(resized_file)):
            os.makedirs(os.path.dirname(resized_file))

        original = Image.fromarray(original_image)
        original.save(original_file)
        resized = Image.fromarray(resized_image)
        resized.save(resized_file)
