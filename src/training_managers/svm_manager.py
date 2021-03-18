"""Model for using Support Vector Machines to make predictions based on data

"""
from configparser import ConfigParser, SectionProxy
import operator
from os import path
import os
import math
import multiprocessing
from typing import List, Union, Dict, Tuple, Any

from sklearn import svm
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

CONSUMER_ID = 'SVM_Manager'
_ENABLED_CONFIGURATION_IDENTIFIER = 'enabled'
_OVERWRITE_EXISTING_CONFIG_ID = 'overwrite existing'
_EXAMPLE_COMBINATION_FACTOR_IDENTIFIER = 'Periods Per Example'
_TDP_BLOCK_LENGTH_IDENTIFIER = "trend deterministic data provider block length"

_CONFIGURABLE_IDENTIFIERS = [_ENABLED_CONFIGURATION_IDENTIFIER, _OVERWRITE_EXISTING_CONFIG_ID,
                             _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER, _TDP_BLOCK_LENGTH_IDENTIFIER]

_CONFIGURATION_DEFAULTS = ['False', 'False', '1', '2520']


def _insert_into_best_model_array(best_model_array: List[Tuple[Any, float]], model, accuracy):
    best_model_array.append((model, accuracy))
    best_model_array.sort(key=operator.itemgetter(1))
    return best_model_array[1:]


def create_svm(input_data, target_data: Union[np.ndarray, List[str]],
               val_input_data: np.ndarray,
               val_target_data: Union[np.ndarray, List[str]]) -> List[svm.SVC]:
    best_models: List[Tuple[Any, float]] = [(None, 0)]
    divisive_factor = 100
    for i in range(1, divisive_factor):
        for k in range(1, divisive_factor//10):
            curr_svm = svm.SVC(kernel='poly', C=i / divisive_factor, degree=1,
                               tol=1e-3, gamma=k*5/(divisive_factor//10))
            curr_svm.fit(input_data, target_data)
            accuracy = test_svm_accuracy(val_input_data, val_target_data, curr_svm)
            best_models = _insert_into_best_model_array(best_models, curr_svm, accuracy)
    return [x[0] for x in best_models]


def test_svm_accuracy(
        val_input_data: np.ndarray,
        val_target_data: Union[np.ndarray, List[str]],
        model: svm.SVC) -> float:
    predictions = model.predict(val_input_data)
    num_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == val_target_data[i]:
            num_correct += 1
    return num_correct / len(predictions)


def handle_data(ticker, training_data, out_dir, overwrite_model, combined_examples=22):
    model_file_path = out_dir + f"{path.sep}{ticker}" + "_{0}.svm"
    x, y, x_dates, y_dates = training_data
    combined_x = np.zeros((len(x) - combined_examples + 1, len(x[0]) * combined_examples))
    for i in range(len(x) - combined_examples + 1):
        examples = x[i:i + combined_examples]
        examples = examples.flatten()
        combined_x[i] = examples

    y = ['Trend Upward' if np.argmax(y[i]) == 1 else "Trend Downward" for i in range(len(y))]

    validation_split = .2
    validation_examples = math.floor(validation_split * len(combined_x))
    valid_x = combined_x[-validation_examples:]
    valid_x_dates = x_dates[-validation_examples:]
    valid_y = y[-validation_examples:]
    valid_y_dates = y_dates[-validation_examples:]
    y = y[:-validation_examples]
    y_dates = y_dates[:-validation_examples]
    combined_x = combined_x[:-validation_examples]
    x_dates = x_dates[:-validation_examples]
    models = create_svm(combined_x, y, valid_x, valid_y)
    for i in range(len(models)):
        fp = model_file_path.format(str(i))
        if path.exists(fp) and not overwrite_model:
            continue
        model = models[i]
        model_accuracy = test_svm_accuracy(valid_x, valid_y, model)
        model_training_accuracy = test_svm_accuracy(combined_x, y, model)
        logger.logger.log(logger.OUTPUT, f"Writing model to {fp}")
        with open(fp, 'wb') as open_file:
            pickle.dump(model, open_file)
        with open(fp.replace('.svm', '.training_info'), 'w') as open_file:
            open_file.write(f"{model_accuracy} {model_training_accuracy}")


def predict_data(ticker, model_dir, prediction_data, combined_examples=22):
    model_path = model_dir + path.sep + f"{ticker}_0.svm"
    if not path.exists(model_path):
        logger.logger.log(logger.WARNING, f"No model exists to make predictions on data from ticker {ticker}."
                                          f"Skipping prediction generation for this stock.")
        return None
    x, y, x_dates, y_dates, unknown_x, unknown_dates = prediction_data

    combined_x = np.zeros((len(x) - combined_examples + 1, len(x[0]) * combined_examples))
    for i in range(len(x) - combined_examples + 1):
        examples = x[i:i + combined_examples]
        examples = examples.flatten()
        combined_x[i] = examples

    y = ['Trend Upward' if np.argmax(y[i]) == 1 else "Trend Downward" for i in range(len(y))]
    y = np.array(y[-132:])
    y_dates = y_dates[-132:]

    model_files = os.listdir(model_dir)
    model_files = [model_dir + path.sep + x for x in model_files if x.startswith(f"{ticker}_") and x.endswith('.svm')]
    predictions = []
    accuracies = []
    for model_path in model_files:
        try:
            with open(model_path, 'rb') as open_file:
                model: svm.SVC = pickle.load(open_file)
        except pickle.UnpicklingError or FileNotFoundError:
            logger.logger.log(logger.NON_FATAL_ERROR, f"Failed to open and unpickle {model_path}."
                                                      f"Skipping prediction generation for this model")
            continue
        prediction_input_data = combined_x[-132:]
        generated_predictions = model.predict(prediction_input_data)
        prediction_dates = x_dates[-len(generated_predictions):]
        correct_predictions = 0
        prediction_info_out_format = "Prediction Date: {0}\nPrediction Input Data: {1}\n" \
                                     "Predicted Date: {2}\nPrediction: {3}\nPrediction Status: {4}\n\n"
        with open(model_path.replace('.svm', '.predict_info'), 'w') as open_file:
            status_history = ""
            for i in range(len(y)):
                prediction_status = "I"
                if generated_predictions[i] == y[i]:
                    correct_predictions += 1
                    prediction_status = "C"
                info_out = prediction_info_out_format.format(prediction_dates[i], prediction_input_data[i],
                                                             y_dates[i], generated_predictions[i], prediction_status)
                status_history += prediction_status
                open_file.write(info_out)
            open_file.write(status_history)
        accuracy = correct_predictions / len(y)
        accuracies.append(accuracy)

        predictions.append(model.predict(unknown_x)[-1])
    return ticker, predictions, accuracies


def string_serialize_predictions(predictions: Dict[str, Tuple[List[str], List[float]]]) -> str:
    ret_str = ""
    for ticker, prediction_data in predictions.items():
        actual_predictions, observed_accuracies = prediction_data
        ret_str += f"Predictions for {ticker}\n"
        for i in range(len(actual_predictions)):
            actual_prediction = actual_predictions[i]
            observed_accuracy = observed_accuracies[i]
            ret_str += f"{actual_prediction} was theorized with an observed accuracy of {observed_accuracy}\n"
    return ret_str


def export_predictions(predictions, output_dir) -> None:
    exportation_columns = []
    for ticker, prediction_data in predictions.items():
        actual_predictions, observed_accuracies = prediction_data
        for i in range(len(actual_predictions)):
            actual_prediction = actual_predictions[i]
            observed_accuracy = observed_accuracies[i]
            exportation_columns.append((ticker, actual_prediction, observed_accuracy))
    csv_exportation.export_predictions(exportation_columns, output_dir + path.sep + 'svm.csv')


class SvmManager(data_provider_registry.DataConsumerBase):

    def __init__(self):
        super(SvmManager, self).__init__()
        self._overwrite_existing = False
        self._combined_examples_factor = 22
        configurable_registry.config_registry.register_configurable(self)

    def consume_data(self, data, passback, output_dir):
        out_dir = output_dir + path.sep + 'svm_models'
        if not path.exists(out_dir):
            os.mkdir(out_dir)
        exec_options = config_parser_singleton.read_execution_options()
        max_processes = exec_options[1]
        max_processes = multiprocessing.cpu_count() if max_processes == -1 else max_processes
        with multiprocessing.Pool(max_processes) as pool:
            open_jobs = []
            for ticker, training_data in data.items():
                job = pool.apply_async(handle_data,
                                       [ticker, training_data, out_dir, self._overwrite_existing],
                                       {'combined_examples': self._combined_examples_factor})
                open_jobs.append(job)
            for job in tqdm.tqdm(open_jobs):
                job.get()

    def predict_data(self, data, passback, in_model_dir):
        model_dir = in_model_dir + path.sep + 'svm_models'
        if not path.exists(model_dir):
            raise FileNotFoundError("Model storage directory for SVM prediction does not exist. Please run"
                                    "Model Creation Main without the prediction flag set to True, and with the"
                                    "SVM Manager's Enabled config to True to create models.")
        predictions = {}
        for ticker, prediction_data in data.items():
            prediction_tuple = predict_data(ticker, model_dir, prediction_data,
                                            combined_examples=self._combined_examples_factor)
            if prediction_tuple is not None:
                ticker, actual_prediction, accuracy = prediction_tuple
            else:
                continue
            predictions[ticker] = (actual_prediction, accuracy)
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
                data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID,
                self,
                [block_length],
                data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID,
                prediction_string_serializer=string_serialize_predictions,
                data_exportation_function=export_predictions,
                keyword_args={'ema_period': [5, 10, 15]}
            )

    def write_default_configuration(self, section: "SectionProxy"):
        for i in range(len(_CONFIGURABLE_IDENTIFIERS)):
            if not _CONFIGURABLE_IDENTIFIERS[i] in section:
                section[_CONFIGURABLE_IDENTIFIERS[i]] = _CONFIGURATION_DEFAULTS[i]


consumer = SvmManager()
