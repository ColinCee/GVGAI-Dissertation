import keras.backend as K
import tensorflow as tf
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dense, Flatten, Lambda
from keras.layers.merge import Add
from keras.models import Model, Sequential
from keras.optimizers import Adam


class Network:
    dqn, ddqn, dueling = range(3)

    def __init__(self, type, input_shape, num_actions):
        self.learning_rate = 0.00025
        self.input_shape = input_shape
        self.num_actions = num_actions

        if type == 0:
            self.primary_network = self.deep_q_network()
            self.target_network = None

        if type == 2:
            self.primary_network = self.dueling_network()
            self.target_network = self.dueling_network()

        self.compile_networks()
        print(self.primary_network.summary())

    def deep_q_network(self):
        network = Sequential()
        network.add(
            Conv2D(32, kernel_size=8, strides=4, input_shape=self.input_shape, data_format="channels_last",
                   activation='relu'))
        network.add(Conv2D(64, kernel_size=4, strides=2, activation='relu'))
        network.add(Conv2D(64, kernel_size=3, strides=1, activation='relu'))
        network.add(Flatten())
        network.add(Dense(512, activation='relu'))
        network.add(Dense(self.num_actions, activation='linear'))
        return network

    def double_deep_q_network(self):
        pass

    def dueling_network(self):
        inputs = Input(shape=self.input_shape)
        net = Conv2D(32, kernel_size=8, strides=4,
                     activation='relu')(inputs)
        net = Conv2D(64, kernel_size=4, strides=2,
                     activation='relu')(net)
        net = Conv2D(64, kernel_size=3, strides=1,
                     activation='relu')(net)
        net = Flatten()(net)
        advt = Dense(256, activation='relu')(net)
        advt = Dense(self.num_actions)(advt)
        value = Dense(256, activation='relu')(net)
        value = Dense(1)(value)
        # now to combine the two streams
        advt = Lambda(lambda advt: advt - tf.reduce_mean(
            advt, axis=-1, keep_dims=True))(advt)
        value = Lambda(lambda value: tf.tile(value, [1, self.num_actions]))(value)
        final = Add()([value, advt])
        model = Model(
            inputs=inputs,
            outputs=final)

        return model

    def mean_Q(self, y_true, y_pred):
        return K.mean(y_pred)

    def compile_networks(self):
        # clip the gradients so that any major differences don't affect the network too much, this may have implications
        # to have fast the network converges
        self.primary_network.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, clipnorm=1., decay=1e-4),
                                     metrics=[self.mean_Q])
        if self.target_network is not None:
            self.target_network.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, clipnorm=1., decay=1e-4),
                                        metrics=[self.mean_Q])
