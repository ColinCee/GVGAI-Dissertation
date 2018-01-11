import random
import os

import numpy as np
from collections import deque

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
import datetime
from Types import ACTIONS


class Brain():

    def __init__(self, input_shape, available_actions):
        self.weight_backup = "weight_backup.h5"
        self.input_shape = input_shape
        self.available_actions = available_actions
        self.memory = deque(maxlen=100000)
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.exploration_rate = 1.0
        self.exploration_min = 0.05
        self.exploration_decay = 0.995
        self.sample_batch_size = 128
        self.model = Sequential()

        self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        # Hidden layers
        self.model.add(Conv2D(24, kernel_size=8, strides=4, input_shape=self.input_shape, activation='relu'))
        self.model.add(Conv2D(32, kernel_size=4, strides=2, activation='relu'))
        self.model.add(Conv2D(48, kernel_size=3, strides=1, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dense(len(self.available_actions), activation='linear')) # Output Layer
        self.model.compile(loss='mae', optimizer=Adam(lr=self.learning_rate))
        print(self.model.summary())

        now = datetime.datetime.now()
        tb_callback = TensorBoard(log_dir='./Graph/' + now.strftime("%I %M"),
                                  write_graph=True, write_images=True)
        tb_callback.set_model(self.model)
        self.callbacks = [tb_callback]

    def save_model(self, episodes):
        dstfolder = "game-snapshots/Episode {}/".format(episodes);
        if not os.path.isdir(dstfolder):
            os.makedirs(dstfolder)  # create all directories, raise an error if it already exists
        self.model.save(os.path.join(dstfolder,self.weight_backup))

    def load_model(self):
        if os.path.isfile(self.weight_backup):
            self.model.load_weights(self.weight_backup)
            self.exploration_rate = 1

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.sample_batch_size:
            return
        sample_batch = random.sample(self.memory, self.sample_batch_size)
        for state, action_string, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(self.get_state_for_NN(next_state))[0])
            target_f = self.model.predict(self.get_state_for_NN(state))

            action_index = self.get_index_of_action(action_string)
            target_f[0][action_index] = target
            self.model.fit(x=self.get_state_for_NN(state), y=target_f,
                           batch_size=self.sample_batch_size, epochs=1,
                           verbose=0, callbacks=self.callbacks)

        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def get_action(self, state, available_actions):
        if np.random.rand() <= self.exploration_rate:
            index = random.randrange(len(available_actions))
            return available_actions[index]

        act_values = self.model.predict(self.get_state_for_NN(state))
        best_action = np.argmax(act_values[0])

        while self.available_actions[best_action] not in available_actions:
            del act_values[0][best_action]
            best_action = np.argmax(act_values[0])

        return self.available_actions[best_action]

    def get_index_of_action(self, action_string):
        for k, v in enumerate(self.available_actions):
            if action_string == v:
                return k

    # Keras expects a 4D array (batch_size, x, y, z)
    # Our image is just a 3D array of (x, y, z)
    def get_state_for_NN(self, state):
        return np.expand_dims(state, axis=0)
