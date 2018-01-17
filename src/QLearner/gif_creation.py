import os

import imageio

original_path = "C:/Users/Colin/Desktop/Backup/Episode 3/original"
resized_path = "C:/Users/Colin/Desktop/Backup/Episode 3/resized"

output_path = "C:/Users/Colin/Desktop/Backup/Episode 3/"

original_frames = []
resized_frames = []

for i in range(1, 275):
    original_filename = os.path.join(original_path, "Step - {}.png".format(i))
    resized_filename = os.path.join(resized_path, "Step - {}.png".format(i))

    original_frames.append(imageio.imread(original_filename))
    resized_frames.append(imageio.imread(resized_filename))

imageio.mimsave(output_path + "original.gif", original_frames, fps=20)
imageio.mimsave(output_path + "resized.gif", resized_frames, fps=20)
