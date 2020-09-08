"""Model for using Support Vector Machine Regression to make predictions based on data

"""
from configparser import ConfigParser, SectionProxy
from os import path
import os
import math
import multiprocessing
from typing import List, Union, Dict, Tuple

from sklearn import svm
from sklearn import exceptions
from sklearn.utils import _testing
import numpy as np
import pickle
import tqdm

from general_utils.config import config_util
from general_utils.config import config_parser_singleton
from general_utils.data_storage_classes import comparison_placeholder
from general_utils.exportation import csv_exportation
from general_utils.logging import logger
from data_providing_module import configurable_registry
from data_providing_module import data_provider_registry
from data_providing_module.data_providers import data_provider_static_names

CONSUMER_ID = 'SVM_Manager'
_ENABLED_CONFIGURATION_IDENTIFIER = 'enabled'
_OVERWRITE_EXISTING_CONFIG_ID = 'overwrite existing'
_EXAMPLE_COMBINATION_FACTOR_IDENTIFIER = 'Periods Per Example'
_TDP_BLOCK_LENGTH_IDENTIFIER = "trend deterministic data provider block length"

_CONFIGURABLE_IDENTIFIERS = [_ENABLED_CONFIGURATION_IDENTIFIER, _OVERWRITE_EXISTING_CONFIG_ID,
                             _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER, _TDP_BLOCK_LENGTH_IDENTIFIER]

_CONFIGURATION_DEFAULTS = ['False', 'False', '1', '2520']


@_testing.ignore_warnings(category=exceptions.ConvergenceWarning)
def _create_poly_svm(input_data, target_data, val_input_data, val_target_data,
                     degree_range: slice, c_divisive_factor: int = 1000):
    least_average_error = comparison_placeholder.ComparisonPlaceholder()
    least_error_model = None
    for j in range(degree_range.start, degree_range.stop):
        for i in range(1, c_divisive_factor):
            curr_svm = svm.SVR(kernel='poly', degree=j, C=i/c_divisive_factor, max_iter=1000)
            curr_svm.fit(input_data, target_data)
            average_error, error_stdev = test_svm_accuracy(val_input_data, val_target_data, curr_svm)
            average_error += error_stdev
            if average_error < least_average_error:
                least_error_model = curr_svm
                least_average_error = average_error
                logger.logger.log(logger.INFORMATION, f"Least error of {least_average_error} "
                                                      f"was achieved with parameters "
                                                      f"[poly, C={i/1000}, degree={j}, tol=1e-2]")
    return least_error_model, least_average_error


@_testing.ignore_warnings(category=exceptions.ConvergenceWarning)
def _create_non_poly_svm(input_data, target_data, val_input_data, val_target_data,
                         kernel: str, c_divisive_factor: int = 1000):
    least_average_error = comparison_placeholder.ComparisonPlaceholder()
    least_error_model = None
    for i in range(1, c_divisive_factor):

        curr_svm = svm.SVR(kernel=kernel, C=i / c_divisive_factor, max_iter=1000)
        curr_svm.fit(input_data, target_data)
        average_error, error_stdev = test_svm_accuracy(val_input_data, val_target_data, curr_svm)
        average_error += error_stdev
        if average_error < least_average_error:
            least_error_model = curr_svm
            least_average_error = average_error
            logger.logger.log(logger.INFORMATION, f"Least error of {least_average_error} "
                                                  f"was achieved with parameters "
                                                  f"[{kernel}, C={i / 1000}, tol=1e-2]")
    return least_error_model, least_average_error


def _remove_inf_and_nan(input_data: np.ndarray):
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            if np.isinf(input_data[i][j]) or np.isnan(input_data[i][j]):
                input_data[i][j] = -1


def create_svm(input_data: np.ndarray, target_data: np.ndarray,
               val_input_data: np.ndarray,
               val_target_data: np.ndarray) -> svm.SVR:
    _remove_inf_and_nan(input_data)
    _remove_inf_and_nan(val_input_data)

    least_error_model, least_average_error = _create_poly_svm(
        input_data, target_data, val_input_data, val_target_data, slice(1, 5), c_divisive_factor=1000
    )

    kernels = ["rbf", 'sigmoid', 'linear']

    for kernel in kernels:
        model, avg_err = _create_non_poly_svm(
            input_data, target_data, val_input_data, val_target_data, kernel, c_divisive_factor=1000
        )
        if avg_err < least_average_error:
            least_error_model = model
            least_average_error = avg_err
    least_error_model.max_iter = 1000000
    least_error_model.fit(input_data, target_data)
    return least_error_model


def test_svm_accuracy(
        val_input_data: np.ndarray,
        val_target_data: Union[np.ndarray, List[str]],
        model: svm.SVR) -> Tuple[float, float]:
    predictions = model.predict(val_input_data)
    errors = [abs(val_target_data[i] - predictions[i]) for i in range(len(predictions))]

    return np.mean(errors), np.std(errors)


def handle_data(ticker, training_data, out_dir, overwrite_model, combined_examples=22):
    model_file_path = out_dir + f"{path.sep}{ticker}.svm"
    if path.exists(model_file_path) and not overwrite_model:
        return
    logger.logger.log(logger.OUTPUT, f"Training model for {ticker}")
    x, y = training_data
    x = x.T
    combined_x = np.zeros((len(x) - combined_examples + 1, len(x[0]) * combined_examples))
    for i in range(len(x) - combined_examples + 1):
        examples = x[i:i + combined_examples]
        examples = examples.flatten()
        combined_x[i] = examples

    validation_split = .2
    validation_examples = math.floor(validation_split * len(combined_x))
    valid_x = combined_x[-validation_examples:]
    valid_y = y[-validation_examples:]
    y = y[:validation_examples]
    combined_x = combined_x[:validation_examples]
    model = create_svm(combined_x, y, valid_x, valid_y)
    model_mean_err, model_stdev_err = test_svm_accuracy(valid_x, valid_y, model)
    model_mean_err_training, model_stdev_err_training = test_svm_accuracy(combined_x, y, model)
    logger.logger.log(logger.OUTPUT, f"Writing model to {model_file_path}")
    with open(model_file_path, 'wb') as open_file:
        pickle.dump(model, open_file)
    with open(model_file_path.replace('.svm', '.training_info'), 'w') as open_file:
        open_file.write(f"{model_mean_err}, {model_stdev_err} {model_mean_err_training}, {model_stdev_err_training}")


def predict_data(ticker, model_dir, prediction_data, combined_examples=22):
    model_path = model_dir + path.sep + f"{ticker}.svm"
    if not path.exists(model_path):
        logger.logger.log(logger.WARNING, f"No model exists to make predictions on data from ticker {ticker}."
                                          f"Skipping prediction generation for this stock.")
        return None
    x, y = prediction_data
    x = x.T

    _remove_inf_and_nan(x)

    combined_x = np.zeros((len(x) - combined_examples + 1, len(x[0]) * combined_examples))
    for i in range(len(x) - combined_examples + 1):
        examples = x[i:i + combined_examples]
        examples = examples.flatten()
        combined_x[i] = examples

    y = np.array(y[-66:])
    try:
        with open(model_path, 'rb') as open_file:
            model: svm.SVR = pickle.load(open_file)
    except pickle.UnpicklingError or FileNotFoundError:
        logger.logger.log(logger.NON_FATAL_ERROR, f"Failed to open and unpickle {model_path}."
                                                  f"Skipping prediction generation for this stock")
        return None
    generated_predictions = model.predict(combined_x[-67:])
    average_error = 0
    for i in range(len(y)):
        average_error += abs(generated_predictions[i] - y[i])

    accuracy = average_error / len(y)
    return ticker, generated_predictions[-1], accuracy


def string_serialize_predictions(predictions: Dict[str, Tuple[str, float]]) -> str:
    ret_str = ""
    for ticker, prediction_data in predictions.items():
        actual_prediction, observed_accuracy = prediction_data
        ret_str += (f"Predictions for {ticker}\n"
                    f"{actual_prediction} was theorized with an observed average error of {observed_accuracy}\n")
    return ret_str


class SvmRegressionManager(data_provider_registry.DataConsumerBase):

    def __init__(self):
        super(SvmRegressionManager, self).__init__()
        self._overwrite_existing = False
        self._combined_examples_factor = 1
        configurable_registry.config_registry.register_configurable(self)

    def consume_data(self, data, passback, output_dir):
        out_dir = output_dir + path.sep + 'svm_regression_models'
        if not path.exists(out_dir):
            os.mkdir(out_dir)
        execution_options = config_parser_singleton.read_execution_options()
        max_processes = int(execution_options[1])
        max_processes = multiprocessing.cpu_count() if max_processes == -1 else max_processes
        with multiprocessing.Pool(max_processes) as pool:
            open_jobs = []
            for ticker, training_data in data.items():
                open_jobs.append(pool.apply_async(handle_data,
                                                  [ticker, training_data, out_dir, self._overwrite_existing],
                                                  {'combined_examples': self._combined_examples_factor}))
            for job in tqdm.tqdm(open_jobs):
                job.get()

    def predict_data(self, data, passback, in_model_dir):
        model_dir = in_model_dir + path.sep + 'svm_regression_models'
        if not path.exists(model_dir):
            raise FileNotFoundError("Model storage directory for SVM prediction does not exist. Please run"
                                    "Model Creation Main without the prediction flag set to True, and with the"
                                    "SVM Manager's Enabled config to True to create models.")
        predictions = {}
        for ticker, prediction_data in data.items():
            ticker, actual_prediction, average_error = predict_data(ticker, model_dir, prediction_data,
                                                                    combined_examples=self._combined_examples_factor)
            predictions[ticker] = (actual_prediction, average_error)
        return predictions

    def load_configuration(self, parser: "ConfigParser"):
        section = config_util.create_type_section(parser, self)
        for identifier in _CONFIGURABLE_IDENTIFIERS:
            if not parser.has_option(section.name, identifier):
                self.write_default_configuration(section)

        enabled = parser.getboolean(section.name, _ENABLED_CONFIGURATION_IDENTIFIER)
        self._overwrite_existing = parser.getboolean(section.name, _OVERWRITE_EXISTING_CONFIG_ID)
        self._combined_examples_factor = parser.getint(section.name, _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER)
        block_length = parser.getint(section.name, _TDP_BLOCK_LENGTH_IDENTIFIER)
        if enabled:
            data_provider_registry.registry.register_consumer(
                data_provider_static_names.CLOSING_PRICE_REGRESSION_BLOCK_PROVIDER_ID,
                self,
                [block_length],
                data_provider_static_names.CLOSING_PRICE_REGRESSION_BLOCK_PROVIDER_ID,
                prediction_string_serializer=string_serialize_predictions
                # ,data_exportation_function=export_predictions
            )

    def write_default_configuration(self, section: "SectionProxy"):
        for i in range(len(_CONFIGURABLE_IDENTIFIERS)):
            if not _CONFIGURABLE_IDENTIFIERS[i] in section:
                section[_CONFIGURABLE_IDENTIFIERS[i]] = _CONFIGURATION_DEFAULTS[i]


consumer = SvmRegressionManager()
