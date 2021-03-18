from configparser import ConfigParser, SectionProxy
from os import path
import os
import multiprocessing
from typing import List, Union, Dict, Tuple, Any

import numpy as np

from general_utils.config import config_util
from general_utils.config import config_parser_singleton
from data_providing_module import configurable_registry
from data_providing_module import data_provider_registry
from data_providing_module.data_providers import data_provider_static_names

CONSUMER_ID = 'Similarity Statistics Manager'
_ENABLED_CONFIGURATION_IDENTIFIER = 'enabled'
_EXAMPLE_COMBINATION_FACTOR_IDENTIFIER = 'Periods Per Example'
_TDP_BLOCK_LENGTH_IDENTIFIER = "trend deterministic data provider block length"
_TOP_N_HIGHLIGHTED_PERIOD = "Number of Top Highlighted Periods"

_CONFIGURABLE_IDENTIFIERS = [_ENABLED_CONFIGURATION_IDENTIFIER, _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER,
                             _TDP_BLOCK_LENGTH_IDENTIFIER, _TOP_N_HIGHLIGHTED_PERIOD]

_CONFIGURATION_DEFAULTS = ['False', '22', '2520', '8']


def _convert_pattern_to_key(pattern: str):
    return pattern.count('U')


def predict_data(ticker, out_dir, prediction_data, combined_examples=22):
    out_folder = out_dir + path.sep + f"{ticker}" + path.sep
    out_path_format = out_folder + f"{ticker}.stat"
    if not path.exists(out_folder):
        os.mkdir(out_folder)
    x, y, x_dates, y_dates, unknown_x, unknown_dates = prediction_data
    combined_x = np.zeros((len(x) - combined_examples + 1, len(x[0]) * combined_examples))
    for i in range(len(x) - combined_examples + 1):
        examples = x[i:i + combined_examples]
        examples = examples.flatten()
        combined_x[i] = examples
    if isinstance(y[0], np.ndarray):
        y = [np.argmax(y[i]) for i in range(len(y))]
    y = y[combined_examples - 1:]
    y_dates = y_dates[combined_examples - 1:]
    base_example = np.append(x[-combined_examples + len(unknown_x):], unknown_x)

    pattern_dict = {}
    total_dist = 0
    pattern_length = 5

    for i in range(len(combined_x)-combined_examples+1):
        dist = 0
        for j in range(len(combined_x[i])):
            base_dist = (base_example[j] - combined_x[i][j])
            if j > 0:
                # penalize example if movement direction is different from the base example
                base_ex_dir = base_example[j] - base_example[j - 1]
                combined_x_dir = combined_x[i][j] - combined_x[i][j - 1]
                base_ex_dir = 1 if abs(base_ex_dir) == base_ex_dir else 0
                combined_x_dir = 1 if abs(combined_x_dir) == combined_x_dir else 0
                if not base_ex_dir == combined_x_dir:
                    base_dist *= 2
            dist += base_dist ** 2
        dist **= .5
        if i+pattern_length >= len(combined_x):
            break
        last_price = combined_x[i][-1]
        pattern_x = combined_x[i+pattern_length]
        pattern = ''
        for j in range(len(pattern_x)):
            if pattern_x[j] >= last_price:
                pattern += 'U'
            else:
                pattern += 'D'
        if pattern not in pattern_dict:
            pattern_dict[pattern] = 0
        pattern_dict[pattern] += dist
        total_dist += dist

    patterns: List[Tuple[str, float]] = []
    for pattern, distance in pattern_dict.items():
        patterns.append((pattern, distance))
    patterns = sorted(patterns, key=_convert_pattern_to_key)
    percentage_format = "{0:.2f}%"
    output_record_format = "{0}:{1}\n"
    with open(out_path_format) as out_file:
        for pattern, distance in patterns:
            historical_similarity = 1 - (distance / total_dist)
            out_record = output_record_format.format(pattern, percentage_format.format(historical_similarity*100))
            out_file.write(out_record)


    # scan through all previous examples, checking their similarity to the base example.
    # Keep track of the largest difference. This becomes 0% similarity.
    # Assuming all previous examples were kept track of, the coding will work as follows
    # For the next x days after the end of the previous example (referred to as x_end), track whether
    # the price at each of those days are above or below x_end. For a day that it is above x_end, a
    # U will be added to the code for the example. When it isn't, a D will be added.
    # Then add %similarity/100 to the total similarity seen and to the tracked code storage dictionary.
    # All that is left is turning all tracked similarities into percentages using the total similarity.

    # output should be ordered by number of Us in the code.


class SimilarityStatisticsManager(data_provider_registry.DataConsumerBase):

    def __init__(self):
        super(SimilarityStatisticsManager, self).__init__()
        configurable_registry.config_registry.register_configurable(self)
        self._combined_examples_factor = 22
        self._num_similar_regions = 8

    def consume_data(self, data, passback, output_dir):
        """Unused method as this manager does not need to be trained."""
        pass

    def predict_data(self, data, passback, in_model_dir):
        out_dir = in_model_dir + path.sep + 'similarity_analysis'
        if not path.exists(out_dir):
            os.mkdir(out_dir)
        exec_options = config_parser_singleton.read_execution_options()
        max_processes = exec_options[1]
        max_processes = multiprocessing.cpu_count() if max_processes == -1 else max_processes
        with multiprocessing.Pool(max_processes) as pool:
            open_jobs = []
            for ticker, prediction_data in data.items():
                open_jobs.append(pool.apply_async(
                    predict_data,
                    [ticker, out_dir, prediction_data],
                    {'combined_examples': self._combined_examples_factor,
                     'num_similar_regions': self._num_similar_regions}
                ))
            for job in open_jobs:
                job.get()

    def load_configuration(self, parser: "ConfigParser"):
        section = config_util.create_type_section(parser, self)
        for identifier in _CONFIGURABLE_IDENTIFIERS:
            if not parser.has_option(section.name, identifier):
                self.write_default_configuration(section)

        enabled = parser.getboolean(section.name, _ENABLED_CONFIGURATION_IDENTIFIER)
        self._combined_examples_factor = parser.getint(section.name, _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER)
        self._num_similar_regions = parser.getint(section.name, _TOP_N_HIGHLIGHTED_PERIOD)
        block_length = parser.getint(section.name, _TDP_BLOCK_LENGTH_IDENTIFIER)
        if enabled:
            data_provider_registry.registry.register_consumer(
                data_provider_static_names.PERCENTAGE_CHANGE_BLOCK_PROVIDER_ID,
                self,
                [block_length],
                data_provider_static_names.PERCENTAGE_CHANGE_BLOCK_PROVIDER_ID,
                prediction_string_serializer=None,
                data_exportation_function=None,
                keyword_args={'percentage_changes': False,
                              'trend_lookahead': 0}
            )

    def write_default_configuration(self, section: "SectionProxy"):
        for i in range(len(_CONFIGURABLE_IDENTIFIERS)):
            if not _CONFIGURABLE_IDENTIFIERS[i] in section:
                section[_CONFIGURABLE_IDENTIFIERS[i]] = _CONFIGURATION_DEFAULTS[i]


consumer = SimilarityStatisticsManager()