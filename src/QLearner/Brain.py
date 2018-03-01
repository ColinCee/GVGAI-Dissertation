import datetime
import os
import random

import numpy as np
from keras.callbacks import TensorBoard

from Network import Network
from Replay import Replay, Sample


class Brain:

    def __init__(self, input_shape, available_actions):
        self.input_shape = input_shape
        self.weight_backup = "weight_backup.h5"
        self.available_actions = available_actions
        self.gamma = 0.99
        self.exploration_rate = 1.0
        self.exploration_min = 0.1
        self.episodes_until_exp_rate_min = 1000
        self.batch_size = 32
        self.network = Network(Network.dqn, self.input_shape, len(available_actions))
        self.replay = Replay(memory_size=100000)
        self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        now = datetime.datetime.now()
        tb_callback = TensorBoard(log_dir='./graph/' + now.strftime("%d %b - %H.%M"),
                                  write_graph=False)
        self.callbacks = [tb_callback]

    def save_model(self, filename):
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))  # create all directories, raise an error if it already exists
        self.network.primary_network.save_weights(os.path.join(filename))

    def load_model(self, filename):
        if os.path.isfile(filename):
            self.network.primary_network.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
            print("Successfully loaded weights!")

    def reduce_exploration_rate(self):
        if self.exploration_rate > self.exploration_min:
            # This will make it so that the exploration rate decays over x episodes until it reaches min
            self.exploration_rate -= (1 - self.exploration_min) / self.episodes_until_exp_rate_min

    def remember(self, state, action, reward, next_state, done):
        self.replay.add_sample(state, action, reward, next_state, done)

    def train(self):
        inputs, targets = self.get_uniform_batch()
        self.network.primary_network.fit(x=inputs, y=targets,
                                         batch_size=self.batch_size,
                                         verbose=0,
                                         callbacks=self.callbacks)

    def get_uniform_batch(self):
        inputs = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        targets = np.zeros((self.batch_size, len(self.available_actions)))
        counter = 0

        data: Sample
        for i in range(self.batch_size):
            data = self.replay.get_sample()
            # Always need to calculate this forward pass
            q1_current_state = self.network.primary_network.predict(
                Brain.transform_input_for_single_sample(data.state))

            if not data.done:
                # Calculate the target depending on if the game has finished or not
                q1_next_state = self.network.primary_network.predict(
                    Brain.transform_input_for_single_sample(data.next_state))
                q2_next_state = self.network.target_network.predict(
                    Brain.transform_input_for_single_sample(data.next_state))
                target = data.reward + self.gamma * q2_next_state[0][np.argmax(q1_next_state[0])]
            else:
                target = data.reward

            action_index = self.get_index_of_action(data.action_string)
            q1_current_state[0][action_index] = target
            inputs[counter] = data.state
            targets[counter] = q1_current_state[0]
            counter += 1

        return inputs, targets

    def get_action(self, state, available_actions):
        if np.random.rand() <= self.exploration_rate:
            index = random.randrange(len(available_actions))
            return available_actions[index]

        act_values = self.network.primary_network.predict(self.transform_input_for_single_sample(state))
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
