from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.activations import relu
from keras.optimizers import Adam
from keras import losses
from typing import Tuple, Callable


def build_dqn(lr, n_actions, input_dims, fc1_dims):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=8, strides=4, activation='relu',
                     input_shape=(*input_dims,), data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=4, strides=2, activation='relu',
                     data_format='channels_first'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=1, activation='relu',
                     data_format='channels_first'))
    model.add(Flatten())
    model.add(Dense(fc1_dims, activation='relu'))
    model.add(Dense(n_actions))

    model.compile(optimizer=Adam(lr=lr), loss='mean_squared_error')

    return model


class DeepQNetwork:

    def __init__(self, learning_rate: float, n_actions: int, input_dimensions: Tuple[int, int, int],
                 fc1_dims: int = 512, build_network_function: Callable = build_dqn):
        self.learning_rate = learning_rate
        self.n_actions = n_actions
        self.input_dimensions = input_dimensions
        self.fc1_dims = fc1_dims
        self.__build_network(build_network_function)

    def __build_network(self, func: Callable):
        self.model = func(self.learning_rate, self.n_actions, self.input_dimensions, self.fc1_dims)

    def predict(self, x):
        return self.model.predict(x)

    def train(self, x, y):
        return self.model.train_on_batch(x, y)

    def store_model(self, filepath: str, include_optimizer: bool = True):
        self.model.save(filepath, include_optimizer=include_optimizer)

    def load_model(self, filepath: str):
        self.model.load_weights(filepath)

