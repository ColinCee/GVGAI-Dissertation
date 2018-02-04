import datetime
import os
import random
from collections import deque

import keras.backend as K
import numpy as np
from Replay import Replay
from Sample import Sample
from keras.callbacks import TensorBoard
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


class Brain():

    def __init__(self, available_actions):
        self.weight_backup = "weight_backup.h5"
        self.available_actions = available_actions
        self.input_shape = (55, 150, 4)
        self.learning_rate = 0.00025
        self.gamma = 0.99
        self.exploration_rate = 1.0
        self.exploration_min = 0.1
        self.episodes_until_exp_rate_min = 1000
        self.batch_size = 32
        self.primary_network = Sequential()
        self.target_network = Sequential()
        self.replay = Replay(memory_size=100000)
        self.PER_alpha = 0.6
        self.PER_epsilon = 0.01
        self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # Primary network
        self.primary_network.add(
            Conv2D(32, kernel_size=8, strides=4, input_shape=self.input_shape, data_format="channels_last",
                   activation='relu'))
        self.primary_network.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
        self.primary_network.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
        self.primary_network.add(Flatten())
        self.primary_network.add(Dense(512, activation='relu'))
        self.primary_network.add(Dense(len(self.available_actions), activation='linear'))  # Output Layer
        # Clip gradients, set metrics
        self.primary_network.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, clipnorm=1., decay=1e-4),
                                     metrics=[self.mean_Q])
        print(self.primary_network.summary())

        # Target Network
        self.target_network.add(
            Conv2D(32, kernel_size=8, strides=4, input_shape=self.input_shape, data_format="channels_last",
                   activation='relu'))
        self.target_network.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
        self.target_network.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
        self.target_network.add(Flatten())
        self.target_network.add(Dense(512, activation='relu'))
        self.target_network.add(Dense(len(self.available_actions), activation='linear'))  # Output Layer
        # Clip gradients, set metrics
        self.target_network.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, clipnorm=1., decay=1e-4),
                                    metrics=[self.mean_Q])

        now = datetime.datetime.now()
        tb_callback = TensorBoard(log_dir='./graph/' + now.strftime("%d %b - %H.%M"),
                                  write_graph=False)
        self.callbacks = [tb_callback]

    def mean_Q(self, y_true, y_pred):
        return K.mean(y_pred)

    def save_model(self, filename):
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))  # create all directories, raise an error if it already exists
        self.primary_network.save(os.path.join(filename))

    def load_model(self, filename):
        if os.path.isfile(filename):
            self.primary_network.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
            print("Successfully loaded weights!")

    def update_target_network(self):
        # Save the primary network weights
        backup = os.path.join("QLearner/model-backup", self.weight_backup)
        self.save_model(backup)
        if os.path.isfile(backup):
            self.target_network.load_weights(backup)
            print("Updating target network...")

    def reduce_exploration_rate(self):
        if self.exploration_rate > self.exploration_min:
            # This will make it so that the exploration rate decays over x episodes until it reaches min
            self.exploration_rate -= (1 - self.exploration_min) / self.episodes_until_exp_rate_min

    def remember(self, state, action, reward, next_state, done):
        data = Sample(state, action, reward, next_state, done)
        self.replay.add_sample(data)

    def train(self):
        inputs, targets = self.get_prioritized_batch()
        self.primary_network.fit(x=inputs, y=targets,
                                 batch_size=self.batch_size,
                                 verbose=0,
                                 callbacks=self.callbacks)

    def get_prioritized_batch(self):
        sample_batch = []
        unique_idx = set()
        inputs = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        targets = np.zeros((self.batch_size, len(self.available_actions)))
        counter = 0

        data: Sample
        for i in range(self.batch_size):
            idx, priority, data = self.replay.get_sample()
            while type(data) is not Sample or idx in unique_idx:
                idx, priority, data = self.replay.get_sample()

            #print("idx: {} - p: {}".format(idx,priority))
            sample_batch.append((idx, priority, data))
            unique_idx.add(idx)
            # Calculate the target depending on if the game has finished or not
            if not data.done:
                next_q = self.target_network.predict(Brain.transform_input_for_single_sample(data.next_state))
                target = data.reward + self.gamma * np.amax(next_q[0])
            else:
                target = data.reward

            target_Q = self.target_network.predict(Brain.transform_input_for_single_sample(data.state))
            action_index = self.get_index_of_action(data.action_string)
            target_Q[0][action_index] = target
            inputs[counter] = data.state
            targets[counter] = target_Q[0]
            # Set new priority
            error = abs(target_Q[0][action_index] - target)
            sample_batch[counter] = (idx, (error + self.PER_epsilon) ** self.PER_alpha, data)
            counter += 1

        self.update_priorities(sample_batch)
        return inputs, targets

    def update_priorities(self, sample_batch):
        for idx, priority, sample in sample_batch:
            self.replay.update_sample(idx, priority)

    def get_action(self, state, available_actions):
        if np.random.rand() <= self.exploration_rate:
            index = random.randrange(len(available_actions))
            return available_actions[index]

        act_values = self.primary_network.predict(self.transform_input_for_single_sample(state))
        best_action = np.argmax(act_values[0])

        while self.available_actions[best_action] not in available_actions:
            del act_values[0][best_action]
            best_action = np.argmax(act_values[0])

        return self.available_actions[best_action]

    def get_index_of_action(self, action_string):
        for k, v in enumerate(self.available_actions):
            if action_string == v:
                return k
        assert 1 == 0, "Could not find action"

    # Keras expects a 4D array (batch_size, x, y, z)
    # Our image is just a 3D array of (x, y, z)
    @staticmethod
    def transform_input_for_single_sample(state):
        return np.expand_dims(state, axis=0)
