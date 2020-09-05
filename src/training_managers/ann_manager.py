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
from general_utils.logging import logger
from general_utils.keras import callbacks
from general_utils.keras import suppression
from data_providing_module import data_provider_registry
from data_providing_module.data_providers import data_provider_static_names


suppression.suppress_tf_deprecation_messages()
suppression.suppress_tf_warnings_and_info()


_ENABLED_CONFIGURATION_IDENTIFIER = 'enabled'
CONSUMER_IDENTIFIER = 'ANN_Manager'
_TDP_BLOCK_LENGTH_IDENTIFIER = 'trend deterministic data provider block length'
_OVERWRITE_EXISTING_MODELS_CONFIG_ID = 'overwrite existing'


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


def handle_data(ticker, training_data, out_dir, overwrite_model):
    model_file_path = out_dir + f"{path.sep}{ticker}.ann"
    if path.exists(model_file_path) and not overwrite_model:
        return
    x, y = training_data
    x = x.T
    combined_examples = 22
    combined_x = np.zeros((len(x) - combined_examples + 1, len(x[0]) * combined_examples))
    for i in range(len(x) - combined_examples + 1):
        examples = x[i:i + combined_examples]
        examples = examples.flatten()
        combined_x[i] = examples

    model = create_ann(len(combined_x[0]), [5], 2)
    validation_split = .2
    validation_examples = math.floor(validation_split * len(combined_x))
    valid_x = combined_x[-validation_examples:]
    valid_y = y[-validation_examples:]
    y = y[:validation_examples]
    combined_x = combined_x[:validation_examples]
    model_callback = callbacks.HighestAccuracyModelStorageCallback(model, out_dir + f"{path.sep}{ticker}.ann")
    model_hist = model.fit(combined_x, y, epochs=3000, validation_data=(valid_x, valid_y),
                           verbose=0, callbacks=[model_callback])


def predict_data(ticker, model_dir, prediction_data):
    model_path = model_dir + path.sep + f"{ticker}.ann"
    if not path.exists(model_path):
        logger.logger.log(logger.WARNING, f"No model exists to make predictions on data from ticker {ticker}."
                                          f"Skipping prediction generation")
        return None
    x, y = prediction_data
    x = x.T
    combined_examples = 22
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
        data_provider_registry.registry.register_consumer(
            data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID,
            self,
            [self._default_tdp_block_length],
            data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID,
            prediction_string_serializer=string_serialize_predictions
        )
        self._overwite_existing = False

    def consume_data(self, data, passback, output_dir):
        open_threads = []
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            out_dir = output_dir + path.sep + "ann_models"
            if not path.exists(out_dir):
                os.mkdir(out_dir)
            for ticker, training_data in data.items():
                open_threads.append(pool.apply_async(handle_data, [ticker, training_data,
                                                                   out_dir, self._overwite_existing]))
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
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            working_threads = []
            for ticker, prediction_data in data.items():
                working_threads.append(pool.apply_async(predict_data, [ticker, model_dir, prediction_data]))
            for worker in tqdm.tqdm(working_threads):
                result = worker.get()
                if result is not None:
                    ticker, actual_prediction, accuracy = result
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
                    data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID,
                    prediction_string_serializer=string_serialize_predictions
                )
        self._overwite_existing = parser.getboolean(section.name, _OVERWRITE_EXISTING_MODELS_CONFIG_ID)

    def write_default_configuration(self, section: "SectionProxy"):
        section[_ENABLED_CONFIGURATION_IDENTIFIER] = 'False'
        section[_TDP_BLOCK_LENGTH_IDENTIFIER] = str(self._default_tdp_block_length)
        section[_OVERWRITE_EXISTING_MODELS_CONFIG_ID] = 'False'


consumer = AnnManager()
