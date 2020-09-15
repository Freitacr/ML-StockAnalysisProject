"""Model for using Gaussian Naive Bayes Models to make predictions based on data

"""
from configparser import ConfigParser, SectionProxy
from os import path
import os
import math
import multiprocessing
from typing import List, Union, Dict, Tuple

from sklearn import naive_bayes
import numpy as np
import pickle
import tqdm

from general_utils.config import config_util
from general_utils.config import config_parser_singleton
from general_utils.exportation import csv_exportation
from general_utils.logging import logger
from data_providing_module import configurable_registry
from data_providing_module import data_provider_registry
from data_providing_module.data_providers import data_provider_static_names

CONSUMER_ID = 'GNB_Manager'
_ENABLED_CONFIGURATION_IDENTIFIER = 'enabled'
_OVERWRITE_EXISTING_CONFIG_ID = 'overwrite existing'
_EXAMPLE_COMBINATION_FACTOR_IDENTIFIER = 'Periods Per Example'
_TDP_BLOCK_LENGTH_IDENTIFIER = "trend deterministic data provider block length"

_CONFIGURABLE_IDENTIFIERS = [_ENABLED_CONFIGURATION_IDENTIFIER, _OVERWRITE_EXISTING_CONFIG_ID,
                             _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER, _TDP_BLOCK_LENGTH_IDENTIFIER]

_CONFIGURATION_DEFAULTS = ['False', 'False', '1', '2520']


def create_naive_bayes_model(input_data, target_data) -> naive_bayes.GaussianNB:
    ret_model = naive_bayes.GaussianNB()
    ret_model.fit(input_data, target_data)
    return ret_model


def test_naive_bayes(val_input_data, val_target_data, model: naive_bayes.GaussianNB) -> float:
    predicted = model.predict(val_input_data)
    num_correct = 0
    for i in range(len(predicted)):
        if predicted[i] == val_target_data[i]:
            num_correct += 1
    return num_correct / len(predicted)


def handle_bayesian_model_creation(ticker, training_data, out_dir, overwrite_model, combined_examples=1):
    model_file_path = out_dir + f"{path.sep}{ticker}.gnb"
    if path.exists(model_file_path) and not overwrite_model:
        return
    x, y, _, _ = training_data
    x = x.T
    combined_x = np.zeros((len(x) - combined_examples + 1, len(x[0]) * combined_examples))
    for i in range(len(x) - combined_examples + 1):
        examples = x[i:i + combined_examples]
        examples = examples.flatten()
        combined_x[i] = examples

    y = ['Trend Upward' if np.argmax(y[i]) == 1 else "Trend Downward" for i in range(len(y))]

    validation_split = .1

    validation_examples = math.floor(validation_split * len(combined_x))
    valid_x = combined_x[-validation_examples:]
    valid_y = y[-validation_examples:]
    y = y[:-validation_examples]
    combined_x = combined_x[:-validation_examples]
    model = create_naive_bayes_model(combined_x, y)
    model_accuracy = test_naive_bayes(valid_x, valid_y, model)
    model_training_accuracy = test_naive_bayes(combined_x, y, model)
    logger.logger.log(logger.INFORMATION, f"Writing model to {model_file_path}")
    with open(model_file_path, 'wb') as open_file:
        pickle.dump(model, open_file)
    with open(model_file_path.replace('.gnb', '.training_info'), 'w') as open_file:
        open_file.write(f"{model_accuracy} {model_training_accuracy}")


def predict_using_models(ticker, model_dir, prediction_data, combined_examples=1):
    model_path = model_dir + path.sep + f"{ticker}.gnb"
    if not path.exists(model_path):
        logger.logger.log(logger.WARNING, f"No model exists to make predictions on data from ticker {ticker}."
                                          f"Skipping prediction generation for this stock.")
        return None
    x, y, _, _ = prediction_data
    x = x.T

    combined_x = np.zeros((len(x) - combined_examples + 1, len(x[0]) * combined_examples))
    for i in range(len(x) - combined_examples + 1):
        examples = x[i:i + combined_examples]
        examples = examples.flatten()
        combined_x[i] = examples

    y = ['Trend Upward' if np.argmax(y[i]) == 1 else "Trend Downward" for i in range(len(y))]
    y = np.array(y[-132:])
    try:
        with open(model_path, 'rb') as open_file:
            model: naive_bayes.GaussianNB = pickle.load(open_file)
    except pickle.UnpicklingError or FileNotFoundError:
        logger.logger.log(logger.NON_FATAL_ERROR, f"Failed to open and unpickle {model_path}."
                                                  f"Skipping prediction generation for this stock")
        return None
    generated_predictions = model.predict(combined_x[-133:])
    correct_predictions = 0
    for i in range(len(y)):
        if generated_predictions[i] == y[i]:
            correct_predictions += 1
    accuracy = correct_predictions / len(y)
    return ticker, generated_predictions[-1], accuracy


def string_serialize_predictions(predictions: Dict[str, Tuple[str, float]]) -> str:
    ret_str = ""
    for ticker, prediction_data in predictions.items():
        actual_prediction, observed_accuracy = prediction_data
        ret_str += (f"Predictions for {ticker}\n"
                    f"{actual_prediction} was theorized with an observed accuracy of {observed_accuracy}\n")
    return ret_str


def export_predictions(predictions, output_dir) -> None:
    exportation_columns = []
    for ticker, prediction_data in predictions.items():
        actual_prediction, observed_accuracy = prediction_data
        exportation_columns.append((ticker, actual_prediction, observed_accuracy))
    csv_exportation.export_predictions(exportation_columns, output_dir + path.sep + 'gnb.csv')


class GNBTrainingManager(data_provider_registry.DataConsumerBase):

    def __init__(self):
        super(GNBTrainingManager, self).__init__()
        self._overwrite_existing = False
        self._periods_per_example = 1
        configurable_registry.config_registry.register_configurable(self)

    def consume_data(self, data, passback, output_dir):
        out_dir = output_dir + path.sep + 'gnb_models'
        if not path.exists(out_dir):
            os.mkdir(out_dir)
        for ticker, training_data in tqdm.tqdm(data.items()):
            handle_bayesian_model_creation(ticker, training_data, out_dir, self._overwrite_existing,
                                           self._periods_per_example)

    def predict_data(self, data, passback, in_model_dir):
        model_dir = in_model_dir + path.sep + 'gnb_models'
        if not path.exists(model_dir):
            raise FileNotFoundError("Model storage directory for Gaussian Naive Bayes prediction does not exist. "
                                    "Please run Model Creation Main without the prediction flag set to True, "
                                    "and with the GNB Manager's Enabled config to True to create models.")
        predictions = {}
        for ticker, prediction_data in data.items():
            ticker, actual_prediction, accuracy = predict_using_models(ticker, model_dir, prediction_data,
                                                                       combined_examples=self._periods_per_example)
            predictions[ticker] = (actual_prediction, accuracy)
        return predictions

    def load_configuration(self, parser: "ConfigParser"):
        section = config_util.create_type_section(parser, self)
        for identifier in _CONFIGURABLE_IDENTIFIERS:
            if not parser.has_option(section.name, identifier):
                self.write_default_configuration(section)

        enabled = parser.getboolean(section.name, _ENABLED_CONFIGURATION_IDENTIFIER)
        self._overwrite_existing = parser.getboolean(section.name, _OVERWRITE_EXISTING_CONFIG_ID)
        self._periods_per_example = parser.getint(section.name, _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER)
        block_length = parser.getint(section.name, _TDP_BLOCK_LENGTH_IDENTIFIER)
        if enabled:
            data_provider_registry.registry.register_consumer(
                data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID,
                self,
                [block_length],
                data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID,
                prediction_string_serializer=string_serialize_predictions,
                data_exportation_function=export_predictions,
                keyword_args={'ema_period': [10, 15, 20]}
            )

    def write_default_configuration(self, section: "SectionProxy"):
        for i in range(len(_CONFIGURABLE_IDENTIFIERS)):
            if not _CONFIGURABLE_IDENTIFIERS[i] in section:
                section[_CONFIGURABLE_IDENTIFIERS[i]] = _CONFIGURATION_DEFAULTS[i]


consumer = GNBTrainingManager()
