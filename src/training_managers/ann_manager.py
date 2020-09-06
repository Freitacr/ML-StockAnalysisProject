"""Module for using standard artificial neural networks to make predictions based on data.

"""

# TODO Create ANN predictor setup to use the Trend Deterministic Block Provider to predict
#   a single day in advance as a proof of concept.

from configparser import ConfigParser, SectionProxy
from os import path
import os
import math
import multiprocessing
from typing import List

from keras import layers
from keras import optimizers
from keras import losses
import keras
import numpy as np
import tqdm

from general_utils.config import config_util
from general_utils.config import config_parser_singleton
from general_utils.logging import logger
from general_utils.keras import callbacks
from general_utils.keras import suppression
from data_providing_module import data_provider_registry
from data_providing_module import configurable_registry
from data_providing_module.data_providers import data_provider_static_names


suppression.suppress_tf_deprecation_messages()
suppression.suppress_tf_warnings_and_info()


_ENABLED_CONFIGURATION_IDENTIFIER = 'enabled'
CONSUMER_IDENTIFIER = 'ANN_Manager'
_TDP_BLOCK_LENGTH_IDENTIFIER = 'trend deterministic data provider block length'
_OVERWRITE_EXISTING_MODELS_CONFIG_ID = 'overwrite existing'
_TREND_LOOKAHEAD_IDENTIFIER = 'trend lookahead'
_EXAMPLE_COMBINATION_FACTOR_IDENTIFIER = 'Periods Per Example'
_EMA_PERIODS_IDENTIFIER = "EMA Calculation Periods"

_CONFIGRABLE_IDIENTIFIERS = [_ENABLED_CONFIGURATION_IDENTIFIER, _TDP_BLOCK_LENGTH_IDENTIFIER,
                             _OVERWRITE_EXISTING_MODELS_CONFIG_ID, _TREND_LOOKAHEAD_IDENTIFIER,
                             _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER, _EMA_PERIODS_IDENTIFIER]


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


def handle_data(ticker, training_data, out_dir, overwrite_model, trend_lookahead=1, combined_examples=22):
    model_file_path = out_dir + f"{path.sep}{ticker}_{trend_lookahead}.ann"
    if path.exists(model_file_path) and not overwrite_model:
        return
    x, y = training_data
    x = x.T
    combined_x = np.zeros((len(x) - combined_examples + 1, len(x[0]) * combined_examples))
    for i in range(len(x) - combined_examples + 1):
        examples = x[i:i + combined_examples]
        examples = examples.flatten()
        combined_x[i] = examples

    model = create_ann(len(combined_x[0]), [20], 2)
    validation_split = .2
    validation_examples = math.floor(validation_split * len(combined_x))
    valid_x = combined_x[-validation_examples:]
    valid_y = y[-validation_examples:]
    y = y[:validation_examples]
    combined_x = combined_x[:validation_examples]
    model_callback = callbacks.HighestAccuracyModelStorageCallback(model, out_dir +
                                                                   f"{path.sep}{ticker}_{trend_lookahead}.ann")
    model_hist = model.fit(combined_x, y, epochs=3000, validation_data=(valid_x, valid_y),
                           verbose=0, callbacks=[model_callback])


def predict_data(ticker, model_dir, prediction_data, trend_lookahead=1, combined_examples=22):
    model_path = model_dir + path.sep + f"{ticker}_{trend_lookahead}.ann"
    if not path.exists(model_path):
        logger.logger.log(logger.WARNING, f"No model exists to make predictions on data from ticker {ticker}."
                                          f"Skipping prediction generation")
        return None
    x, y = prediction_data
    x = x.T
    combined_x = np.zeros((len(x) - combined_examples + 1, len(x[0]) * combined_examples))
    for i in range(len(x) - combined_examples + 1):
        examples = x[i:i + combined_examples]
        examples = examples.flatten()
        combined_x[i] = examples

    y = np.array(y[-66:])
    try:
        model: keras.Model = keras.models.load_model(model_path)
    except OSError:
        logger.logger.log(logger.NON_FATAL_ERROR, f"Failed to open {model_path}. Skipping prediction.")
        return
    generated_predictions = model.predict(combined_x[-67:])
    correct_predictions = 0
    for i in range(len(y)):
        correct = np.argmax(y[i])
        if np.argmax(generated_predictions[i]) == correct:
            correct_predictions += 1
    accuracy = correct_predictions / len(y)
    return ticker, generated_predictions[-1], accuracy


def string_serialize_predictions(predictions) -> str:
    ret_str = ""
    for ticker, prediction_data in predictions.items():
        actual_prediction, observed_accuracy = prediction_data
        predicted = np.argmax(actual_prediction)
        if predicted == 1:
            prediction_str = "Trend Upward"
        else:
            prediction_str = "Trend Downward"
        ret_str += (f"Predictions for {ticker}\n"
                    f"{prediction_str} was theorized with an observed accuracy of {observed_accuracy}\n")
    return ret_str


class AnnManager(data_provider_registry.DataConsumerBase):

    def __init__(self):
        super(AnnManager, self).__init__()
        self._default_tdp_block_length = 252 * 10
        self._trend_lookahead = 1
        self._overwite_existing = False
        self._combined_examples_factor = 22
        self._ema_periods = [10]
        configurable_registry.config_registry.register_configurable(self)

    def consume_data(self, data, passback, output_dir):
        open_threads = []
        _, max_processes = config_parser_singleton.read_execution_options()
        max_processes = multiprocessing.cpu_count() if max_processes == -1 else max_processes
        with multiprocessing.Pool(max_processes) as pool:
            out_dir = output_dir + path.sep + "ann_models"
            if not path.exists(out_dir):
                os.mkdir(out_dir)
            for ticker, training_data in data.items():
                open_threads.append(pool.apply_async(handle_data, [ticker, training_data,
                                                                   out_dir, self._overwite_existing],
                                                     {'trend_lookahead': self._trend_lookahead,
                                                      'combined_examples': self._combined_examples_factor}))
            for t in tqdm.tqdm(open_threads):
                t.get()
                # model_hist = model.fit(combined_x, y, epochs=3000, validation_split=.2,
                #                        verbose=2, callbacks=[model_callback])

    def predict_data(self, data, passback, in_model_dir):
        model_dir = in_model_dir + path.sep + 'ann_models'
        if not path.exists(model_dir):
            raise FileNotFoundError("Model storage directory for ANN prediction does not exist. Please run model "
                                    "creation without the prediction flag set to true to create models used in "
                                    "prediction.")
        predictions = {}
        _, max_processes = config_parser_singleton.read_execution_options()
        max_processes = multiprocessing.cpu_count() if max_processes == -1 else max_processes
        with multiprocessing.Pool(max_processes) as pool:
            working_threads = []
            for ticker, prediction_data in data.items():
                working_threads.append(pool.apply_async(predict_data,
                                                        [ticker, model_dir, prediction_data],
                                                        {'trend_lookahead': self._trend_lookahead,
                                                         'combined_examples': self._combined_examples_factor}))
            for worker in tqdm.tqdm(working_threads):
                result = worker.get()
                if result is not None:
                    ticker, actual_prediction, accuracy = result
                    predictions[ticker] = (actual_prediction, accuracy)
        return predictions

    def load_configuration(self, parser: "ConfigParser"):
        section = config_util.create_type_section(parser, self)
        for identifier in _CONFIGRABLE_IDIENTIFIERS:
            if not parser.has_option(section.name, identifier):
                self.write_default_configuration(section)

        enabled = parser.getboolean(section.name, _ENABLED_CONFIGURATION_IDENTIFIER)
        self._trend_lookahead = parser.getint(section.name, _TREND_LOOKAHEAD_IDENTIFIER)
        self._overwite_existing = parser.getboolean(section.name, _OVERWRITE_EXISTING_MODELS_CONFIG_ID)
        self._combined_examples_factor = parser.getint(section.name, _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER)
        ema_periods = parser.get(section.name, _EMA_PERIODS_IDENTIFIER)
        ema_periods = ema_periods.split(',')
        self._ema_periods = [int(x) for x in ema_periods]
        block_length = parser.getint(section.name, _TDP_BLOCK_LENGTH_IDENTIFIER)
        if enabled:
            data_provider_registry.registry.register_consumer(
                data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID,
                self,
                [block_length],
                data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID,
                {"trend_lookahead": self._trend_lookahead,
                 "ema_period": self._ema_periods},
                prediction_string_serializer=string_serialize_predictions
            )

    def write_default_configuration(self, section: "SectionProxy"):
        section[_ENABLED_CONFIGURATION_IDENTIFIER] = 'False'
        section[_TDP_BLOCK_LENGTH_IDENTIFIER] = str(self._default_tdp_block_length)
        section[_OVERWRITE_EXISTING_MODELS_CONFIG_ID] = 'False'
        section[_TREND_LOOKAHEAD_IDENTIFIER] = '1'
        section[_EXAMPLE_COMBINATION_FACTOR_IDENTIFIER] = '22'
        section[_EMA_PERIODS_IDENTIFIER] = "10"


consumer = AnnManager()
