import datetime
import os
import random
from collections import deque

import keras.backend as K
import numpy as np
from keras.callbacks import TensorBoard
from keras.layers import Dense, Conv2D, Flatten
from keras.models import Sequential
from keras.optimizers import Adam


class Brain():

    def __init__(self, available_actions):
        self.weight_backup = "weight_backup.h5"
        self.input_shape = (55, 150, 4)
        self.available_actions = available_actions
        self.memory = deque(maxlen=100000)
        self.learning_rate = 0.0002
        self.gamma = 0.99
        self.exploration_rate = 1.0
        self.exploration_min = 0.1
        self.exploration_decay = 0.004  # 225 iterations to 0.1
        self.sample_batch_size = 32
        self.primary_network = Sequential()
        self.target_network = Sequential()
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
        self.primary_network.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, clipnorm=1.),
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
        self.primary_network.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, clipnorm=1.),
                                     metrics=[self.mean_Q])

        now = datetime.datetime.now()
        tb_callback = TensorBoard(log_dir='./graph/' + now.strftime("%d %b - %H.%M"),
                                  write_graph=False)
        self.callbacks = [tb_callback]

    def mean_Q(self, y_true, y_pred):
        return K.mean(y_pred)

    def save_model(self, filename):
        if not os.path.isdir(filename):
            os.makedirs(filename)  # create all directories, raise an error if it already exists
        self.primary_network.save(os.path.join(filename, self.weight_backup))

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
            print("Successfully updated target weights!")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.sample_batch_size:
            return

        inputs, targets = self.get_batch()
        self.primary_network.fit(x=inputs, y=targets,
                                 initial_epoch=0,
                                 batch_size=self.sample_batch_size,
                                 verbose=0,
                                 callbacks=self.callbacks)

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate -= self.exploration_decay

    def get_batch(self):
        sample_batch = random.sample(self.memory, self.sample_batch_size)
        inputs = np.zeros((self.sample_batch_size, 55, 150, 4))
        targets = np.zeros((self.sample_batch_size, 3))
        counter = 0

        for state, action_string, reward, next_state, done in sample_batch:

            if not done:
                next_q = self.target_network.predict(self.get_state_for_NN(next_state))
                target = reward + self.gamma * np.amax(next_q[0])
            else:
                target = reward

            target_f = self.target_network.predict(self.get_state_for_NN(state))
            action_index = self.get_index_of_action(action_string)
            target_f[0][action_index] = target
            inputs[counter] = state
            targets[counter] = target_f[0]
            counter += 1
        return inputs, targets

    def get_action(self, state, available_actions):
        if np.random.rand() <= self.exploration_rate:
            index = random.randrange(len(available_actions))
            return available_actions[index]

        act_values = self.primary_network.predict(self.get_state_for_NN(state))
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
    def get_state_for_NN(self, state):
        return np.expand_dims(state, axis=0)
