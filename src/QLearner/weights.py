import cv2
import matplotlib.pyplot as plt

from State import State

plt.rcParams.update({'figure.max_open_warning': 0})
import os
from keras import backend as K
from keras import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense, np
from keras.optimizers import Adam
from CompetitionParameters import CompetitionParameters


# def get_model(input_shape, output_size):
#     model = Sequential()
#     model.add(Conv2D(16, kernel_size=8, strides=4, input_shape=input_shape, activation='relu'))
#     model.add(Conv2D(32, kernel_size=4, strides=2, activation='relu'))
#     model.add(Conv2D(32, kernel_size=3, strides=1, activation='relu'))
#     model.add(Flatten())
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(output_size, activation='linear'))  # Output Layer
#     model.compile(loss='mae', optimizer=Adam(lr=0.001))
#     print(model.summary())
#
#     if os.path.isfile("/weight_backup.h5"):
#         print("Loading weights")
#         model.load_weights("/weight_backup.h5")
#     else:
#         print("Failed to load weights.")
#
#     return model


def get_layer_outputs(model, input):
    test_image = np.expand_dims(input, axis=0)
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    comp_graph = [K.function([model.input] + [K.learning_phase()], [output]) for output in
                  outputs]  # evaluation functions

    # Testing
    layer_outputs_list = [op([test_image, 1.]) for op in comp_graph]
    layer_outputs = []

    for layer_output in layer_outputs_list:
        # print(layer_output[0][0].shape, end='\n-------------------\n')
        layer_outputs.append(layer_output[0][0])

    return layer_outputs


def get_filters_for_layer(layer_number, layer_outputs):
    x_max = layer_outputs[layer_number].shape[0]
    y_max = layer_outputs[layer_number].shape[1]
    n = layer_outputs[layer_number].shape[2]

    L = []
    for i in range(n):
        L.append(np.zeros((x_max, y_max)))

    for i in range(n):
        for x in range(x_max):
            for y in range(y_max):
                L[i][x][y] = layer_outputs[layer_number][x][y][i]

    return L


def plot_filters_for_layer(layer_filters, layer_number, episode_number):
    # Show all the filters
    rows = 8
    cols = 8

    plt.figure()
    plt.suptitle("Episode: {} - Layer number: {}".format(episode_number, layer_number + 1))

    for index, img in enumerate(layer_filters):
        # cv2.imwrite("output.png", img)
        subplot_ax = plt.subplot(rows, cols, index + 1)
        subplot_ax.axis("off")
        im = subplot_ax.imshow(img, interpolation='nearest')

    filename = "filters/Episode {}/Layer {}".format(episode_number, layer_number + 1)
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))

    plt.savefig(filename, bbox_inches='tight')


def get_test_input():
    im1 = State.get_single_frame('QLearner/samples/original/Step - 304.png')
    im2 = State.get_single_frame('QLearner/samples/original/Step - 308.png')
    im3 = State.get_single_frame('QLearner/samples/original/Step - 312.png')
    #im4 = State.get_single_frame('QLearner/samples/original/Step - 316.png')

    images = [im1, im2, im3]
    stacks = np.stack(images, axis=2)

    return stacks


def plot_all_layers(model, episode_number):
    test_image = get_test_input()
    # test_image = get_test_input(CompetitionParameters.SCREENSHOT_FILENAME)

    # Show the original picture
    plt.figure()
    plt.imshow(test_image)
    plt.title("Episode {}".format(episode_number))

    layer_outputs = get_layer_outputs(model, test_image)

    for i in range(3):
        layer_filters = get_filters_for_layer(i, layer_outputs)
        plot_filters_for_layer(layer_filters, i, episode_number)
