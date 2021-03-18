"""Data Provider Implementation Module for constructing data based on the percentage change between periods

"""

import configparser
import datetime

import numpy as np

from data_providing_module.data_providers import data_provider_static_names
from data_providing_module import configurable_registry
from data_providing_module import data_provider_registry
from general_utils.config import config_util
from general_utils.logging import logger
from general_utils.mysql_management.mysql_tables import stock_data_table
from stock_data_analysis_module.data_processing_module.data_retrieval_module import period_data_retriever

_ENABLED_CONFIG_ID = 'enabled'


def _calc_percentage_changes(data: np.ndarray):
    ret_changes = np.zeros((len(data)-1, ))
    for i in range(len(data)-1):
        if data[i] == 0:
            ret_changes[i] = 0
        else:
            ret_changes[i] = ((data[i+1] - data[i]) / data[i]) * 100
    return ret_changes


def _generate_target_data(*args, **kwargs):
    max_rows = args[-2]
    end_date = args[-1]
    data_retriever = period_data_retriever.PeriodDataRetriever(
        [stock_data_table.CLOSING_PRICE_COLUMN_NAME, stock_data_table.HISTORICAL_DATE_COLUMN_NAME],
        end_date
    )
    ret_target_data = {}
    for ticker, _ in data_retriever.data_sources.items():
        ticker_data = data_retriever.retrieve_data(ticker, max_rows=max_rows)
        ticker_data = np.array(ticker_data)
        close = ticker_data[:, 0]
        close = np.array(list(reversed(close)), dtype=np.float32)
        hist_dates = ticker_data[:, 1]
        hist_dates = np.array(list(reversed(hist_dates)))

        if kwargs['percentage_changes']:
            close = _calc_percentage_changes(close)
        else:
            close = close[1:]
        hist_dates = hist_dates[:-1]
        date_keyed_changes = {}
        for i in range(len(hist_dates)):
            date_keyed_changes[hist_dates[i]] = close[i]
        ret_target_data[ticker] = (date_keyed_changes, hist_dates)
    return ret_target_data


def _generate_training_data(*args, **kwargs):
    max_rows = args[-2]
    end_date = args[-1]
    data_retriever = period_data_retriever.PeriodDataRetriever(
        [
            stock_data_table.HIGH_PRICE_COLUMN_NAME,
            stock_data_table.LOW_PRICE_COLUMN_NAME,
            stock_data_table.CLOSING_PRICE_COLUMN_NAME,
            stock_data_table.VOLUME_COLUMN_NAME,
            stock_data_table.HISTORICAL_DATE_COLUMN_NAME
        ],
        end_date
    )
    ret_training_data = {}
    for ticker, _ in data_retriever.data_sources.items():
        ticker_data = data_retriever.retrieve_data(ticker, max_rows=max_rows)
        ticker_data = np.array(ticker_data)

        high = ticker_data[:, 0]
        low = ticker_data[:, 1]
        close = ticker_data[:, 2]
        volume = ticker_data[:, 3]
        hist_dates = ticker_data[:, 4]

        high = np.array(list(reversed(high)), dtype=np.float32)
        low = np.array(list(reversed(low)), dtype=np.float32)
        close = np.array(list(reversed(close)), dtype=np.float32)
        volume = np.array(list(reversed(volume)), dtype=np.float32)
        hist_dates = np.array(list(reversed(hist_dates)))

        high = _calc_percentage_changes(high)
        low = _calc_percentage_changes(low)
        close = _calc_percentage_changes(close)
        volume = _calc_percentage_changes(volume)
        hist_dates = hist_dates[:-1]
        date_keyed_data = {}
        for i in range(len(high)):
            date_keyed_data[hist_dates[i]] = [high[i], low[i], close[i], volume[i]]
        ret_training_data[ticker] = (date_keyed_data, hist_dates)
    return ret_training_data


def _generate_agnostic_data(*args, **kwargs):
    if not args:
        raise ValueError("Expected %d positional argument but received %d" % (1, len(args)))
    data_block_length = args[0]

    if 'percentage_changes' not in kwargs:
        kwargs['percentage_changes'] = True

    ret_blocks = {}
    end_date = datetime.datetime.now().isoformat()[:10].replace('-', '/')
    training_data = _generate_training_data(
        *args, data_block_length + 1, end_date,
        **kwargs
    )
    target_data = _generate_target_data(
        *args, data_block_length + 1, end_date,
        **kwargs
    )
    for ticker in training_data.keys():
        date_keyed_data, training_dates = training_data[ticker]
        date_keyed_changes, target_dates = target_data[ticker]
        trend_lookahead = kwargs['trend_lookahead']
        ret_training = []
        ret_training_dates = []
        ret_target = []
        ret_target_dates = []

        for i in range(len(training_dates)):
            if i + trend_lookahead >= len(training_dates):
                break
            training_date = training_dates[i]
            target_date = target_dates[i+trend_lookahead]

            ret_training.append(date_keyed_data[training_date])
            ret_target.append(date_keyed_changes[target_date])

            ret_training_dates.append(training_date)
            ret_target_dates.append(target_date)

        if 'predict' not in kwargs or not kwargs['predict']:
            ret_blocks[ticker] = (np.array(ret_training), np.array(ret_target),
                                  np.array(ret_training_dates), np.array(ret_target_dates))
        else:
            if trend_lookahead == 0:
                unknown_data_dates = []
            else:
                unknown_data_dates = [x for x in training_dates[-trend_lookahead:]]
            unknown_data = [date_keyed_data[x] for x in unknown_data_dates]
            ret_blocks[ticker] = (np.array(ret_training), np.array(ret_target),
                                  np.array(ret_training_dates), np.array(ret_target_dates),
                                  np.array(unknown_data), np.array(unknown_data_dates))
    return ret_blocks


class PercentageChangeBlockProvider(data_provider_registry.DataProviderBase):

    def generate_data(self, *args, **kwargs):
        return _generate_agnostic_data(*args, **kwargs)

    def generate_prediction_data(self, *args, **kwargs):
        kwargs['predict'] = True
        return _generate_agnostic_data(*args, **kwargs)

    def load_configuration(self, parser: "ConfigParser"):
        section = config_util.create_type_section(parser, self)
        if not parser.has_option(section.name, _ENABLED_CONFIG_ID):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, _ENABLED_CONFIG_ID)
        if enabled:
            data_provider_registry.registry.register_provider(
                data_provider_static_names.PERCENTAGE_CHANGE_BLOCK_PROVIDER_ID, self)

    def write_default_configuration(self, section: "SectionProxy"):
        section[_ENABLED_CONFIG_ID] = 'True'

    def __init__(self):
        super().__init__()
        configurable_registry.config_registry.register_configurable(self)


provider = PercentageChangeBlockProvider()
