"""Module for using standard artificial neural networks to make predictions based on data.

"""

# TODO Create ANN predictor setup to use the Trend Deterministic Block Provider to predict
#   a single day in advance as a proof of concept.

from configparser import ConfigParser, SectionProxy
from os import path
import os
from typing import List

from keras import layers
from keras import optimizers
from keras import losses
from keras import activations
import keras
import numpy as np

from general_utils.config import config_util
from general_utils.logging import logger
from general_utils.keras import callbacks
from data_providing_module import data_provider_registry
from data_providing_module.data_providers import data_provider_static_names


_ENABLED_CONFIGURATION_IDENTIFIER = 'enabled'
CONSUMER_IDENTIFIER = 'ANN_Manager'
_TDP_BLOCK_LENGTH_IDENTIFIER = 'trend deterministic data provider block length'


def create_ann(input_dim: int, hidden_dimensions: List[int], out_dimensions=None
               ) -> keras.Model:
    if out_dimensions is None:
        out_dimensions = [1]
    input_layer = layers.Input((input_dim,))
    previous_layer = input_layer
    for dim in hidden_dimensions:
        layer = layers.Dense(dim, activation='tanh')(previous_layer)
        previous_layer = layer
    # dropout = layers.Dropout(.6)(previous_layer)
    out_layer = layers.Dense(out_dimensions, activation='tanh')(previous_layer)
    out_layer = layers.Activation('softmax')(out_layer)
    model = keras.Model(inputs=input_layer, outputs=out_layer)
    model.compile(optimizers.Adam(lr=1e-5), loss=losses.categorical_crossentropy, metrics=['accuracy'])
    return model


class AnnManager(data_provider_registry.DataConsumerBase):

    def __init__(self):
        super(AnnManager, self).__init__()
        self._default_tdp_block_length = 252 * 10
        data_provider_registry.registry.register_consumer(
            data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID,
            self,
            [self._default_tdp_block_length],
            data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID
        )

    def consume_data(self, data, passback, output_dir):
        out_dir = output_dir + path.sep + "ann_models"
        if not path.exists(out_dir):
            os.mkdir(out_dir)
        for ticker, training_data in data.items():
            x, y_np = training_data
            y = []
            x = x.T
            combined_examples = 22
            combined_x = np.zeros((len(x)-combined_examples+1, len(x[0])*combined_examples))
            for i in range(len(x)-combined_examples+1):
                examples = x[i:i+combined_examples]
                examples = examples.flatten()
                combined_x[i] = examples

            for i in range(combined_examples-1, len(y_np)):
                y.append([0, 1] if y_np[i] == 1 else [1, 0])
            y = np.array(y)
            model = create_ann(len(combined_x[0]), [5], 2)
            model_callback = callbacks.HighestAccuracyModelStorageCallback(model, out_dir + f"{path.sep}{ticker}.ann")
            model_hist = model.fit(combined_x, y, epochs=3000, validation_split=.2,
                                   verbose=2, callbacks=[model_callback])

    def predict_data(self, data, passback, in_model_dir):
        model_dir = in_model_dir + path.sep + 'ann_models'
        if not path.exists(model_dir):
            raise FileNotFoundError("Model storage directory for ANN prediction does not exist. Please run model "
                                    "creation without the prediction flag set to true to create models used in "
                                    "prediction.")
        predictions = {}
        for ticker, prediction_data in data.items():
            model_path = model_dir + path.sep + f"{ticker}.ann"
            if not path.exists(model_path):
                logger.logger.log(logger.WARNING, f"No model exists to make predictions on data from ticker {ticker}."
                                                  f"Skipping prediction generation")
                continue
            x, y_np = prediction_data
            y = []
            x = x.T
            combined_examples = 22
            combined_x = np.zeros((len(x) - combined_examples + 1, len(x[0]) * combined_examples))
            for i in range(len(x) - combined_examples + 1):
                examples = x[i:i + combined_examples]
                examples = examples.flatten()
                combined_x[i] = examples

            for i in range(combined_examples - 1, len(y_np)):
                y.append([0, 1] if y_np[i] == 1 else [1, 0])
            y = np.array(y[-22:])
            model: keras.Model = keras.models.load_model(model_path)
            generated_predictions = model.predict(combined_x[-23:])
            correct_predictions = 0
            for i in range(len(y)):
                correct = np.argmax(y[i])
                if np.argmax(generated_predictions[i]) == correct:
                    correct_predictions += 1
            accuracy = correct_predictions / len(y)
            actual_prediction = 'down' if np.argmax(generated_predictions[-1]) == 0 else 'up'
            predictions[ticker] = (actual_prediction, accuracy)
        return predictions

    def load_configuration(self, parser: "ConfigParser"):
        section = config_util.create_type_section(parser, self)
        if not parser.has_option(section.name, _ENABLED_CONFIGURATION_IDENTIFIER):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, _ENABLED_CONFIGURATION_IDENTIFIER)
        if not enabled:
            data_provider_registry.registry.deregister_consumer(
                data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID, self
            )
        else:
            block_length = parser.getint(section.name, _TDP_BLOCK_LENGTH_IDENTIFIER)
            if block_length != self._default_tdp_block_length:
                data_provider_registry.registry.deregister_consumer(
                    data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID, self
                )
                data_provider_registry.registry.register_consumer(
                    data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID,
                    self,
                    [block_length],
                    data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID
                )

    def write_default_configuration(self, section: "SectionProxy"):
        section[_ENABLED_CONFIGURATION_IDENTIFIER] = 'True'
        section[_TDP_BLOCK_LENGTH_IDENTIFIER] = str(self._default_tdp_block_length)


consumer = AnnManager()
