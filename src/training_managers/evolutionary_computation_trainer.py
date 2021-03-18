"""

"""
from configparser import ConfigParser, SectionProxy
from os import path
import os
from typing import List, Tuple, Any, Optional, Dict

import numpy as np
import tqdm

from general_utils.config import config_util, config_parser_singleton
from general_utils.exportation import csv_exportation
from general_utils.logging import logger
from data_providing_module import configurable_registry, data_provider_registry
from data_providing_module.data_providers import data_provider_static_names
from stock_data_analysis_module.ml_models.evolutionary_computation import TradingPopulation

CONSUMER_ID = "Evolutionary Computation Trainer"
_ENABLED_CONFIGURATION_IDENTIFIER = 'enabled'
_EXAMPLE_COMBINATION_FACTOR_IDENTIFIER = 'Periods Per Example'
_TDP_BLOCK_LENGTH_IDENTIFIER = "trend deterministic data provider block length"
_NUM_EPOCHS_IDENTIFIER = "Number of Epochs"
_NUM_INDIVIDUALS_IDENTIFIER = "Number of Individuals in Evolutionary Population"
_MODEL_CHECKPOINT_EPOCH_INTERVAL_IDENTIFIER = "Model Saving Epoch Interval"
_TRAINING_PERIODS_PER_EXAMPLE_IDENTIFIER = "Days Per Example"
_MUTATION_CHANCE_IDENTIFIER = "Mutation Chance Per Genome"
_MUTATION_MAGNITUDE_IDENTIFIER = "Mutation Magnitude"
_CROSSOVER_CHANCE_IDENTIFIER = "Crossover Chance Per Genome"

_CONFIGURABLE_IDENTIFIERS = [_ENABLED_CONFIGURATION_IDENTIFIER, _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER,
                             _TDP_BLOCK_LENGTH_IDENTIFIER, _NUM_EPOCHS_IDENTIFIER, _NUM_INDIVIDUALS_IDENTIFIER,
                             _MODEL_CHECKPOINT_EPOCH_INTERVAL_IDENTIFIER, _TRAINING_PERIODS_PER_EXAMPLE_IDENTIFIER,
                             _MUTATION_CHANCE_IDENTIFIER, _MUTATION_MAGNITUDE_IDENTIFIER, _CROSSOVER_CHANCE_IDENTIFIER]

_CONFIGURATION_DEFAULTS = ['False', '22', '2520', '100', '100', '5', '5', '.1', '.15', '.5']


def string_serialize_predictions(predictions) -> str:
    ret_str = ""
    ticker_prediction_template = "{}:{}\n"
    individual_prediction_template = "{}:\n\t\tBuy: {}\n\t\tSell: {}\n\t\tAccuracies: {:.2f}, {:.2f}"
    for ticker, data in predictions.items():
        ticker_predictions, accuracies = data
        serialized_individual_predictions = []
        for i in range(len(ticker_predictions)):
            indicate_buy = ticker_predictions[i][0] == 1
            indicate_sell = ticker_predictions[i][1] == 1
            serialized_individual_predictions.append(
                individual_prediction_template.format(i+1, indicate_buy, indicate_sell,
                                                      accuracies[i][0], accuracies[i][1])
            )
        expanded_template = ticker_prediction_template.format(ticker, "\n\t{}" * len(ticker_predictions))
        ret_str += expanded_template.format(*serialized_individual_predictions)
    return ret_str


def export_predictions(predictions, output_dir) -> None:
    out_file = output_dir + path.sep + "ec.csv"
    exportation_columns = []
    for ticker, prediction_data in predictions.items():
        actual_predictions, observed_accuracies = prediction_data
        actual_predictions = np.where(actual_predictions == 1, True, False)
        exportation_columns.append((ticker, "", ""))
        for i in range(len(actual_predictions)):
            exportation_columns.append((",Model:", str(i)))
            exportation_columns.append((",Buy:", str(actual_predictions[i][0])))
            exportation_columns.append((",Buy Accuracy:", str(observed_accuracies[i][0])))
            exportation_columns.append((",Sell:", str(actual_predictions[i][1])))
            exportation_columns.append((",Sell Accuracy:", str(observed_accuracies[i][1])))
    with open(out_file, 'w') as handle:
        for column in exportation_columns:
            handle.write(",".join(column) + '\n')


def prediction_truth_calculation(predictions: List[np.ndarray],
                                 closing_prices: List[float],
                                 num_days_per_prediction: int = 5):
    prediction_entry = Tuple[List[np.ndarray], float, List[List[bool]]]
    prediction_array: List[Optional[prediction_entry]] = [None] * (num_days_per_prediction+1)
    current_index = 0
    ret = []
    for i in range(len(predictions)):
        for j in range(1, len(prediction_array)):
            index = (j + current_index) % len(prediction_array)
            if prediction_array[index] is None:
                continue
            for k in range(len(prediction_array[index][0])):
                prediction, reference_price, prediction_truths = prediction_array[index]
                prediction = prediction[k]
                prediction_truths = prediction_truths[k]
                if reference_price < closing_prices[i]:
                    if prediction[0]:
                        prediction_truths[0] = True
                    if not prediction[1]:
                        prediction_truths[1] = True
                elif reference_price > closing_prices[i]:
                    if not prediction[0]:
                        prediction_truths[0] = True
                    if prediction[1]:
                        prediction_truths[1] = True
        if prediction_array[current_index] is not None:
            prediction_truth = prediction_array[current_index][-1]
            ret.append(prediction_truth)
        prediction_array[current_index] = ([*predictions[i]], closing_prices[i], [[False, False]] * len(predictions[i]))
        current_index += 1
        current_index %= len(prediction_array)
    return ret


def extract_accuracy_from_prediction_truths(prediction_truths: List[List[List[bool]]]):
    ret = np.zeros((len(prediction_truths[0]), len(prediction_truths[0][0])))
    for i in range(len(prediction_truths)):
        for prediction_index, truths in enumerate(prediction_truths[i]):
            for index, truth in enumerate(truths):
                if truth:
                    ret[prediction_index][index] += 1
    ret /= len(prediction_truths)
    return ret


class EvolutionaryComputationManager(data_provider_registry.DataConsumerBase):

    def __init__(self):
        super().__init__()
        configurable_registry.config_registry.register_configurable(self)
        self.__contained_population: Optional[TradingPopulation] = None
        self.__periods_per_example = 5
        self.__num_epochs = 100
        self.__num_individuals = 100
        self.__save_interval = 5
        self.__mutation_chance = .1
        self.__mutation_magnitude = .15
        self.__crossover_chance = .5

    def consume_data(self, data: Dict[str, Tuple[np.ndarray, List[float]]], passback, output_dir):
        out_dir = output_dir + path.sep + 'evolutionary_computation_models'
        if not path.exists(out_dir):
            os.mkdir(out_dir)
        previous_model_file = out_dir + path.sep + "evolution_individuals.ecp"
        if path.exists(previous_model_file):
            self.__contained_population = TradingPopulation((0, 0), 0, 0)
            self.__contained_population.load(previous_model_file)
        else:
            num_indicators = len(data[next(iter(data.keys()))][0])
            input_shape = (num_indicators, self.__periods_per_example)
            self.__contained_population = TradingPopulation(input_shape, 1000, self.__num_individuals,
                                                            self.__mutation_chance, self.__mutation_magnitude,
                                                            self.__crossover_chance)
        consolidated_data: Dict[str, Tuple[np.ndarray, List[float]]] = {}
        for ticker, ticker_data in data.items():
            daily_data, closing_prices = ticker_data
            consolidated_data[ticker] = self.construct_examples(daily_data, closing_prices)
        self.__train_model(consolidated_data, previous_model_file)
        self.__contained_population.save(previous_model_file)

    def __print_best_fitness_by_ticker(self, best_fitness_by_ticker: Dict[str, List[float]]) -> None:
        output_template = "{ticker}:\n\t{:.2f}\n\t{:.2f}\n\t{:.2f}\n"
        for ticker, fitness in best_fitness_by_ticker.items():
            logger.logger.log(logger.INFORMATION, output_template.format(
                ticker=ticker, *fitness
            ))

    def __train_model(self, consolidated_data: Dict[str, Tuple[np.ndarray, List[float]]], previous_model_file: str):
        for i in tqdm.tqdm(range(self.__num_epochs)):
            best_fitness_by_ticker = {}
            for ticker, ticker_data in consolidated_data.items():
                daily_data, closing_prices = ticker_data
                best_fitness = self.__contained_population.train(daily_data, 1, closing_prices)
                best_fitness_by_ticker[ticker] = best_fitness
            self.__print_best_fitness_by_ticker(best_fitness_by_ticker)
            if i % self.__save_interval == 0:
                self.__contained_population.save(previous_model_file)
        self.__contained_population.save(previous_model_file)

    def predict_data(self, data, passback, in_model_dir):
        in_dir = in_model_dir + path.sep + 'evolutionary_computation_models'
        if not path.exists(in_dir):
            raise FileNotFoundError("Model storage directory for EC prediction does not exist. Please run"
                                    "Model Creation Main without the prediction flag set to True, and with the"
                                    "EC Manager's Enabled config to True to create models."
                                    )
        self.__contained_population = TradingPopulation((0, 0), 0, 0)
        self.__contained_population.load(in_dir + path.sep + 'evolution_individuals.ecp')
        consolidated_data: Dict[str, Tuple[np.ndarray, List[float]]] = {}
        for ticker, ticker_data in data.items():
            daily_data, closing_prices = ticker_data
            consolidated_data[ticker] = self.construct_examples(daily_data, closing_prices)
        predictions = {}
        for ticker, prediction_data in consolidated_data.items():
            daily_data, closing_prices = prediction_data
            model_predictions = []
            for i in range(len(daily_data)):
                prediction = self.__contained_population.predict(daily_data[i])
                model_predictions.append(prediction)
            truths = prediction_truth_calculation(model_predictions[:-1], closing_prices)
            accuracies = extract_accuracy_from_prediction_truths(truths)
            prediction = self.__contained_population.predict(daily_data[-1])
            predictions[ticker] = (prediction, accuracies)
        return predictions

    def load_configuration(self, parser: "ConfigParser"):
        section = config_util.create_type_section(parser, self)
        for identifier in _CONFIGURABLE_IDENTIFIERS:
            if not parser.has_option(section.name, identifier):
                self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, _ENABLED_CONFIGURATION_IDENTIFIER)
        self.__periods_per_example = parser.getint(section.name, _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER)
        self.__num_individuals = parser.getint(section.name, _NUM_INDIVIDUALS_IDENTIFIER)
        self.__num_epochs = parser.getint(section.name, _NUM_EPOCHS_IDENTIFIER)
        self.__save_interval = parser.getint(section.name, _MODEL_CHECKPOINT_EPOCH_INTERVAL_IDENTIFIER)
        self.__mutation_chance = parser.getfloat(section.name, _MUTATION_CHANCE_IDENTIFIER)
        self.__mutation_magnitude = parser.getfloat(section.name, _MUTATION_MAGNITUDE_IDENTIFIER)
        self.__crossover_chance = parser.getfloat(section.name, _CROSSOVER_CHANCE_IDENTIFIER)
        block_length = parser.getint(section.name, _TDP_BLOCK_LENGTH_IDENTIFIER)
        if enabled:
            data_provider_registry.registry.register_consumer(
                data_provider_static_names.CLOSING_PRICE_REGRESSION_BLOCK_PROVIDER_ID,
                self,
                [block_length],
                data_provider_static_names.CLOSING_PRICE_REGRESSION_BLOCK_PROVIDER_ID,
                keyword_args={'ema_period': [10, 15, 20]},
                data_exportation_function=export_predictions,
                prediction_string_serializer=string_serialize_predictions
            )

    def write_default_configuration(self, section: "SectionProxy"):
        for i in range(len(_CONFIGURABLE_IDENTIFIERS)):
            if not _CONFIGURABLE_IDENTIFIERS[i] in section:
                section[_CONFIGURABLE_IDENTIFIERS[i]] = _CONFIGURATION_DEFAULTS[i]

    def construct_examples(self, daily_data: np.ndarray, closing_prices: List[float]) -> Tuple[np.ndarray, List[float]]:
        ret_daily_data = np.zeros((
            daily_data.shape[1] - self.__periods_per_example + 1,
            len(daily_data),
            self.__periods_per_example
        ))
        for i in range(self.__periods_per_example, daily_data.shape[1]+1):
            ret_daily_data[i - self.__periods_per_example] = daily_data[:, i - self.__periods_per_example: i]
        return ret_daily_data, closing_prices[self.__periods_per_example-1:]


if "testing" not in os.environ:
    consumer = EvolutionaryComputationManager()
