"""Manager for generating reports detailing how previous periods of similar trends continued moving.

"""
from configparser import ConfigParser, SectionProxy
import operator
from os import path
import os
import multiprocessing
from typing import List, Union, Dict, Tuple, Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import dates
from matplotlib import figure

from general_utils.config import config_util
from general_utils.config import config_parser_singleton
from data_providing_module import configurable_registry
from data_providing_module import data_provider_registry
from data_providing_module.data_providers import data_provider_static_names


CONSUMER_ID = 'Similarity Highlight Manager'
_ENABLED_CONFIGURATION_IDENTIFIER = 'enabled'
_EXAMPLE_COMBINATION_FACTOR_IDENTIFIER = 'Periods Per Example'
_TDP_BLOCK_LENGTH_IDENTIFIER = "trend deterministic data provider block length"
_TOP_N_HIGHLIGHTED_PERIOD = "Number of Top Highlighted Periods"

_CONFIGURABLE_IDENTIFIERS = [_ENABLED_CONFIGURATION_IDENTIFIER, _EXAMPLE_COMBINATION_FACTOR_IDENTIFIER,
                             _TDP_BLOCK_LENGTH_IDENTIFIER, _TOP_N_HIGHLIGHTED_PERIOD]

_CONFIGURATION_DEFAULTS = ['False', '22', '2520', '8']


def _insert_into_best_model_array(best_model_array: List[Tuple[Any, float]], model, accuracy):
    best_model_array.append((model, accuracy))
    best_model_array.sort(key=operator.itemgetter(1), reverse=True)
    return best_model_array[1:]


def predict_data(ticker, out_dir, prediction_data, combined_examples=22, num_similar_regions=8):
    out_folder = out_dir + path.sep + f"{ticker}" + path.sep
    out_path_format = out_folder + f"{ticker}" + "_{0}_{1:.2f}.jpg"
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
    y = y[combined_examples-1:]
    y_dates = y_dates[combined_examples-1:]
    base_example = np.append(x[-combined_examples+len(unknown_x):], unknown_x)
    least_different_periods: List[Tuple[Any, float]] = [(None, np.inf) for _ in range(num_similar_regions)]

    figure_data_formatter = dates.DateFormatter('%d-%m-%y')

    # Skip over the most recent combined examples to avoid having a high similarity due to
    # incorporating most of the base example.
    for i in range(len(combined_x)-combined_examples+1):
        dist = 0
        for j in range(len(combined_x[i])):
            dist += (base_example[j] - combined_x[i][j]) ** 2
        dist **= .5
        least_different_periods = _insert_into_best_model_array(least_different_periods, i, dist)
    fig_index = len(least_different_periods)-1
    for y_index, dist in least_different_periods:
        y_range_start = max(0, y_index-combined_examples)
        y_range = y[y_range_start:y_index+combined_examples]
        y_date_range = list(y_dates[y_range_start: y_index + combined_examples])
        if y[y_index] > y[y_index+1]:
            color = "#ff0000"
        else:
            color = "#00ff00"

        fig: figure.Figure = plt.figure()
        plt.grid(True)
        plt.xlabel("Date Range")
        plt.ylabel("Closing Price")
        axes = fig.axes
        axes[0].xaxis.set_ticks([y_date_range[0], y_dates[y_index], y_date_range[-1]])
        axes[0].xaxis.set_major_formatter(figure_data_formatter)
        axes[0].set_facecolor("#3f3f3f")
        axes[0].xaxis.label.set_color('#afafaf')
        axes[0].tick_params(axis='both', colors='#afafaf')
        axes[0].yaxis.label.set_color('#afafaf')
        plt.plot(y_date_range, y_range, marker='o', color=color)
        plt.savefig(out_path_format.format(fig_index, dist), bbox_inches='tight', facecolor="#0f0f0f")
        fig_index -= 1
        plt.close()

    known_example_base_range = y_dates[-combined_examples+len(unknown_x):]
    base_fig = plt.figure()
    plt.grid(True)
    plt.xlabel("Date Range")
    plt.ylabel("Closing Price")
    axes = base_fig.axes
    axes[0].xaxis.set_ticks([known_example_base_range[0], known_example_base_range[-1]])
    axes[0].xaxis.set_major_formatter(figure_data_formatter)
    axes[0].set_facecolor("#3f3f3f")
    axes[0].xaxis.label.set_color('#afafaf')
    axes[0].tick_params(axis='both', colors='#afafaf')
    axes[0].yaxis.label.set_color('#afafaf')
    plt.plot(known_example_base_range,
             y[-combined_examples+len(unknown_x):], marker='o')
    plt.savefig(out_path_format.format('base', 0.0), bbox_inches='tight', facecolor='#0f0f0f')
    plt.close()


class SimilarityHighlightingManager(data_provider_registry.DataConsumerBase):

    def __init__(self):
        super(SimilarityHighlightingManager, self).__init__()
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


consumer = SimilarityHighlightingManager()
