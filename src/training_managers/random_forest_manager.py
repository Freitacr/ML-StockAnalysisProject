"""Manager for creation and usage of Random Forest Classifiers to make predictions

"""
from configparser import ConfigParser, SectionProxy
import math
import multiprocessing
import operator
from os import path
import os
from typing import List, Union, Dict, Tuple, Any

from sklearn import ensemble
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

CONSUMER_ID = 'Random_Forest_Manager'
_ENABLED_CONFIG_IDENTIFIER = 'enabled'
_OVERWRITE_EXISTING_CONFIG_ID = 'overwrite existing'
_EXAMPLE_COMBINATION_FACTOR_IDENTIFIER = 'Periods Per Example'
_TDP_BLOCK_LENGTH_CONFIG_ID = 'trend deterministic data provider block length'

_CONFIGURABLE_IDENTIFIERS = [_ENABLED_CONFIG_IDENTIFIER, _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER,
                             _OVERWRITE_EXISTING_CONFIG_ID, _TDP_BLOCK_LENGTH_CONFIG_ID]

_CONFIG_DEFAULTS = ['False', '1', 'False', '2520']


def _insert_into_best_model_array(best_model_array: List[Tuple[Any, float]], model, accuracy):
    best_model_array.append((model, accuracy))
    best_model_array.sort(key=operator.itemgetter(1))
    return best_model_array[1:]


def create_random_forest(input_data, target_data, val_input_data, val_target_data
                         ) -> List[ensemble.RandomForestClassifier]:
    highest_accuracy_models: List[Tuple[Any, float]] = [(None, 0)]
    criterion = ['gini', 'entropy']
    for split_criterion in criterion:
        for i in range(2, 6):
            forest_size = 200
            for j in range(1, 6):
                divisive_factor = 5
                samples = None if j / divisive_factor == 1 else j / divisive_factor
                model = ensemble.RandomForestClassifier(n_estimators=forest_size, min_samples_split=i,
                                                        max_samples=samples, criterion=split_criterion)
                model.fit(input_data, target_data)
                accuracy = test_forest_accuracy(val_input_data, val_target_data, model)
                highest_accuracy_models = _insert_into_best_model_array(highest_accuracy_models, model, accuracy)
    return [x[0] for x in highest_accuracy_models]


def test_forest_accuracy(val_input_data, val_target_data, model) -> float:
    predictions = model.predict(val_input_data)
    num_correct = 0
    for i in range(len(predictions)):
        if predictions[i] == val_target_data[i]:
            num_correct += 1
    return num_correct / len(predictions)


def handle_model_creation(ticker, training_data, out_dir, overwrite_model, combined_examples=1):
    model_file_path = out_dir + f"{path.sep}{ticker}" + "_{0}.randomforest"
    x, y, _, _ = training_data
    x = x.T
    combined_x = np.zeros((len(x) - combined_examples + 1, len(x[0]) * combined_examples))
    for i in range(len(x) - combined_examples + 1):
        examples = x[i:i + combined_examples]
        examples = examples.flatten()
        combined_x[i] = examples

    y = [np.argmax(y[i]) for i in range(len(y))]

    validation_split = .1
    validation_examples = math.floor(validation_split * len(combined_x))
    valid_x = combined_x[-validation_examples:]
    valid_y = y[-validation_examples:]
    y = y[:-validation_examples]
    combined_x = combined_x[:-validation_examples]

    models = create_random_forest(combined_x, y, valid_x, valid_y)
    for i in range(len(models)):
        fp = model_file_path.format(str(i))
        if path.exists(fp) and not overwrite_model:
            continue
        model = models[i]
        model_accuracy = test_forest_accuracy(valid_x, valid_y, model)
        model_training_accuracy = test_forest_accuracy(combined_x, y, model)
        logger.logger.log(logger.INFORMATION, f"Writing model to {fp}")
        with open(fp, 'wb') as open_file:
            pickle.dump(model, open_file)
        with open(fp.replace('.randomforest', '.training_info'), 'w') as open_file:
            open_file.write(f"{model_accuracy} {model_training_accuracy}")


def predict_using_models(ticker, model_dir, prediction_data, combined_examples=1):
    model_path = model_dir + path.sep + f"{ticker}_0.randomforest"
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

    y = [np.argmax(y[i]) for i in range(len(y))]
    y = np.array(y[-132:])

    model_files = os.listdir(model_dir)
    model_files = [model_dir + path.sep + x for x in model_files
                   if x.startswith(f"{ticker}_") and x.endswith('.randomforest')]
    predictions = []
    accuracies = []
    for model_path in model_files:
        try:
            with open(model_path, 'rb') as open_file:
                model: ensemble.RandomForestClassifier = pickle.load(open_file)
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
        accuracies.append(accuracy)
        predictions.append(generated_predictions[-1])
    return ticker, predictions, accuracies


def string_serialize_predictions(predictions) -> str:
    ret_str = ""
    for ticker, prediction_data in predictions.items():
        ret_str += f"Predictions for {ticker}\n"
        actual_predictions, observed_accuracies = prediction_data
        for i in range(len(actual_predictions)):
            actual_prediction = actual_predictions[i]
            observed_accuracy = observed_accuracies[i]
            if actual_prediction == 1:
                prediction_str = "Trend Upward"
            else:
                prediction_str = "Trend Downward"
            ret_str += f"{prediction_str} was theorized with an observed accuracy of {observed_accuracy}\n"
    return ret_str


def export_predictions(predictions, output_dir) -> None:
    exportation_columns = []
    for ticker, prediction_data in predictions.items():
        actual_predictions, observed_accuracies = prediction_data
        for i in range(len(actual_predictions)):
            actual_prediction = actual_predictions[i]
            observed_accuracy = observed_accuracies[i]
            if actual_prediction == 1:
                prediction_str = "Trend Upward"
            else:
                prediction_str = "Trend Downward"
            exportation_columns.append((ticker, prediction_str, observed_accuracy))
    csv_exportation.export_predictions(exportation_columns, output_dir + path.sep + 'random_forest.csv')


class RandomForestManager(data_provider_registry.DataConsumerBase):

    def __init__(self):
        super(RandomForestManager, self).__init__()
        configurable_registry.config_registry.register_configurable(self)
        self._overwrite_existing = False
        self._periods_per_example = 1

    def consume_data(self, data, passback, output_dir):
        out_dir = output_dir + path.sep + 'random_forest_models'
        if not path.exists(out_dir):
            os.mkdir(out_dir)
        _, max_processes, _ = config_parser_singleton.read_execution_options()
        max_processes = multiprocessing.cpu_count() if max_processes == -1 else max_processes
        with multiprocessing.Pool(max_processes) as pool:
            tasks = []
            for ticker, training_data in data.items():
                tasks.append(pool.apply_async(handle_model_creation,
                                              [ticker, training_data, out_dir, self._overwrite_existing],
                                              {"combined_examples": self._periods_per_example}))
            for task in tqdm.tqdm(tasks):
                task.get()

    def predict_data(self, data, passback, in_model_dir):
        model_dir = in_model_dir + path.sep + 'random_forest_models'
        if not path.exists(model_dir):
            raise FileNotFoundError("Model storage directory for SVM prediction does not exist. Please run"
                                    "Model Creation Main without the prediction flag set to True, and with the"
                                    "SVM Manager's Enabled config to True to create models.")
        predictions = {}
        for ticker, prediction_data in data.items():
            prediction_tuple = predict_using_models(ticker, model_dir, prediction_data,
                                                    combined_examples=self._periods_per_example)
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

        enabled = parser.getboolean(section.name, _ENABLED_CONFIG_IDENTIFIER)
        self._overwrite_existing = parser.getboolean(section.name, _OVERWRITE_EXISTING_CONFIG_ID)
        self._periods_per_example = parser.getint(section.name, _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER)
        block_length = parser.getint(section.name, _TDP_BLOCK_LENGTH_CONFIG_ID)
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
                section[_CONFIGURABLE_IDENTIFIERS[i]] = _CONFIG_DEFAULTS[i]


consumer = RandomForestManager()

