"""Data Provider Implementation Module for constructing data based on Trend Deterministic Data

Guiding Principles of this implementation are available in the paper
"Predicting stock and stock price index movement using Trend Deterministic Data
Preparation and machine learning techniques" (2014, Patel, Shah, et al)

As with other Data Provider implementations, this provider is not intended to be
instantiated outside of this module. All interactions with this provider are to
be done through the global DataProviderRegistry's registration and data passing systems.

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
from stock_data_analysis_module.indicators import commodity_channel_index
from stock_data_analysis_module.indicators import moving_average
from stock_data_analysis_module.indicators import momentum
from stock_data_analysis_module.indicators import rate_of_change
from stock_data_analysis_module.indicators import stochastic_oscillator

_ENABLED_CONFIG_ID = 'enabled'


def _generate_target_data(*args, **kwargs):
    padded_block_length = args[-2]
    end_date = args[-1]
    data_retriever = period_data_retriever.PeriodDataRetriever(
        [stock_data_table.CLOSING_PRICE_COLUMN_NAME, stock_data_table.HISTORICAL_DATE_COLUMN_NAME],
        end_date
    )
    ret_target_data = {}
    for ticker, _ in data_retriever.data_sources.items():
        ticker_data = data_retriever.retrieve_data(ticker, max_rows=padded_block_length)
        ticker_data = np.array(ticker_data)
        close = ticker_data[:, 0]
        close = np.array(list(reversed(close)), dtype=np.float32)
        hist_dates = ticker_data[:, 1]
        hist_dates = np.array(list(reversed(hist_dates)))

        if not kwargs['trend_strength_labelling']:
            actual_trends = []
            for i in range(1, len(close)):
                if close[i] >= close[i-1]:
                    actual_trends.append([0, 1])
                else:
                    actual_trends.append([1, 0])
        else:
            actual_trends = []
            for i in range(1, len(close)):
                percentage_diff = (close[i] - close[i-1]) / close[i-1]
                if percentage_diff >= .01:
                    actual_trends.append([0, 0, 0, 0, 1])
                elif percentage_diff >= .005:
                    actual_trends.append([0, 0, 0, 1, 0])
                elif percentage_diff >= -.005:
                    actual_trends.append([0, 0, 1, 0, 0])
                elif percentage_diff >= -.01:
                    actual_trends.append([0, 1, 0, 0, 0])
                else:
                    actual_trends.append([1, 0, 0, 0, 0])
        actual_trends = np.array(actual_trends)
        trend_dates = hist_dates[:-1]
        date_keyed_trends = {}
        for i in range(len(trend_dates)):
            date_keyed_trends[trend_dates[i]] = actual_trends[i]
        ret_target_data[ticker] = (date_keyed_trends, np.array(trend_dates))
    return ret_target_data


def _generate_training_data(*args, **kwargs):
    padded_block_length = args[-3]
    end_date = args[-2]
    minimum_examples = args[-1]
    data_retriever = period_data_retriever.PeriodDataRetriever(
        [
            stock_data_table.HIGH_PRICE_COLUMN_NAME,
            stock_data_table.LOW_PRICE_COLUMN_NAME,
            stock_data_table.CLOSING_PRICE_COLUMN_NAME,
            stock_data_table.HISTORICAL_DATE_COLUMN_NAME
        ],
        end_date
    )
    ret_training_data = {}
    for ticker, _ in data_retriever.data_sources.items():
        ticker_data = data_retriever.retrieve_data(ticker, max_rows=padded_block_length)
        ticker_data = np.array(ticker_data)

        high = ticker_data[:, 0]
        low = ticker_data[:, 1]
        close = ticker_data[:, 2]
        hist_dates = ticker_data[:, 3]

        high = np.array(list(reversed(high)), dtype=np.float32)
        low = np.array(list(reversed(low)), dtype=np.float32)
        close = np.array(list(reversed(close)), dtype=np.float32)
        hist_dates = np.array(list(reversed(hist_dates)))

        if len(high) < minimum_examples:
            len_warning = (
                    "Could not process %s into an indicator block, "
                    "needed %d days of trading data but received %d" %
                    (ticker, minimum_examples, len(high))
            )
            logger.logger.log(logger.WARNING, len_warning)
            ret_training_data[ticker] = None, None
            continue

        sma = moving_average.SMA(close, kwargs['sma_period'])
        for i in range(len(sma) - 1):
            sma[i] = 1 if sma[i] <= sma[i + 1] else -1
        sma = sma[:-1]

        wma = moving_average.WMA(close, kwargs['wma_period'])
        for i in range(len(wma) - 1):
            wma[i] = 1 if wma[i] <= wma[i + 1] else -1
        wma = wma[:-1]

        emas = []
        for period in kwargs['ema_period']:
            ema = moving_average.EMA(close, period)
            for i in range(len(ema) - 1):
                ema[i] = 1 if ema[i] <= ema[i + 1] else -1
            emas.append(ema[:-1].flatten())

        macd_ema_1 = moving_average.EMA(close, kwargs['macd_ema_1_period'])
        macd_ema_2 = moving_average.EMA(close, kwargs['macd_ema_2_period'])
        macd_ema_2 = macd_ema_2[-len(macd_ema_1):]
        macd = macd_ema_2 - macd_ema_1
        for i in range(len(macd) - 1):
            macd[i] = 1 if macd[i] <= macd[i + 1] else -1
        macd = macd[:-1]

        price_momentum = momentum.momentum(close, kwargs['momentum_period'])
        for i in range(len(price_momentum) - 1):
            price_momentum[i] = 1 if price_momentum[i] <= price_momentum[i + 1] else -1
        price_momentum = price_momentum[:-1]

        roc = rate_of_change.rate_of_change(close, kwargs['rate_of_change_period'])
        for i in range(len(roc) - 1):
            roc[i] = 1 if roc[i] <= roc[i + 1] else -1
        roc = roc[:-1]

        oscillator = stochastic_oscillator.stochastic_oscillator(close, high,
                                                                 low, kwargs['oscillator_period'])
        for i in range(len(oscillator) - 1):
            if oscillator[i + 1] >= 80:
                oscillator[i] = -1
            elif oscillator[i + 1] <= 20:
                oscillator[i] = 1
            else:
                oscillator[i] = 1 if oscillator[i] <= oscillator[i + 1] else -1
        oscillator = oscillator[:-1]

        cci = commodity_channel_index.commodity_channel_index(close, high,
                                                              low, kwargs['cci_period'])

        for i in range(len(cci) - 1):
            if cci[i + 1] >= 200:
                cci[i] = -1
            elif cci[i + 1] <= -200:
                cci[i] = 1
            else:
                cci[i] = 1 if cci[i] <= cci[i + 1] else -1
        cci = cci[:-1]

        ad_oscillator = stochastic_oscillator.ad_oscillator(close, high, low)
        for i in range(len(ad_oscillator) - 1):
            ad_oscillator[i] = 1 if ad_oscillator[i] <= ad_oscillator[i + 1] else -1
        ad_oscillator = ad_oscillator[:-1]

        stock_data_block = [sma, wma]
        stock_data_block.extend(emas)
        stock_data_block.extend([macd.flatten(), price_momentum, roc, oscillator, ad_oscillator, cci])
        min_len = len(sma)
        for data in stock_data_block:
            if min_len > len(data):
                min_len = len(data)
        for i in range(len(stock_data_block)):
            stock_data_block[i] = stock_data_block[i][-min_len:]

        stock_data_block = np.array(stock_data_block).T

        data_block_dates = hist_dates[-min_len:]

        keyed_data_entries = {}
        for i in range(len(stock_data_block)):
            keyed_data_entries[data_block_dates[i]] = stock_data_block[i]
        ret_training_data[ticker] = (keyed_data_entries, np.array(data_block_dates))
    return ret_training_data


class TrendDeterministicBlockProvider(data_provider_registry.DataProviderBase):

    def _generate_agnostic_data(self, *args, **kwargs):
        if not args:
            raise ValueError("Expected %d positional argument but received %d" % (1, len(args)))
        data_block_length = args[0]
        max_additional_period = 0
        for key, value in self.default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = self.default_kwargs[key]

            if isinstance(value, int):
                if value > max_additional_period:
                    max_additional_period = value
            elif isinstance(value, list):
                for period in value:
                    if period > max_additional_period:
                        max_additional_period = period
        # as most indicators are translated into trend deterministic data by comparison with the
        # previous period's result, and additional period of trading data is required.
        max_additional_period += 1

        padded_data_block_length = max_additional_period + data_block_length

        ret_blocks = {}
        end_date = datetime.datetime.now().isoformat()[:10].replace('-', '/')
        training_data = _generate_training_data(
            *args, padded_data_block_length, end_date, data_block_length//2,
            **kwargs
        )
        target_data = _generate_target_data(*args, padded_data_block_length, end_date, **kwargs)
        for ticker in training_data.keys():
            keyed_data_entries, data_block_dates = training_data[ticker]
            date_keyed_trends, trend_dates = target_data[ticker]

            if keyed_data_entries is None:
                continue

            trend_lookahead = kwargs['trend_lookahead']
            # trend_lookahead += 1

            ret_block = []
            ret_block_dates = []
            ret_trends = []
            ret_trend_dates = []

            for i in range(len(data_block_dates)):
                if i + trend_lookahead >= len(data_block_dates):
                    break
                if data_block_dates[i+trend_lookahead] not in date_keyed_trends:
                    continue
                training_date = data_block_dates[i]
                prediction_date = data_block_dates[i + trend_lookahead]

                ret_block.append(keyed_data_entries[training_date])
                ret_trends.append(date_keyed_trends[prediction_date])

                ret_block_dates.append(training_date)
                ret_trend_dates.append(prediction_date)

            if 'predict' not in kwargs:
                ret_blocks[ticker] = (np.array(ret_block), np.array(ret_trends),
                                      np.array(ret_block_dates), np.array(ret_trend_dates))
            else:
                unknown_data_dates = [x for x in data_block_dates[-trend_lookahead:]]
                unknown_data = [keyed_data_entries[x] for x in unknown_data_dates]
                ret_blocks[ticker] = (np.array(ret_block), np.array(ret_trends),
                                      np.array(ret_block_dates), np.array(ret_trend_dates),
                                      np.array(unknown_data), np.array(unknown_data_dates))
        return ret_blocks

    def generate_data(self, *args, **kwargs):
        """Generates data by converting stock indicators into Trend Deterministic Data

        Generates a dictionary mapping a stock ticker to the data block generated from that
        stock's historical data

        One positional argument is required as noted in the Arguments section, but keyword
        arguments exist for the calculation of the indicators used in data generation.

        Arguments:
            *args:
                data_block_length: int This controls the maximum number of columns returned
                    in the data block for a particular stock. If a stock does not have data
                    for enough periods to satisfy this length, less columns will be returned.
            **kwargs:
                sma_period: int Controls how many periods are considered in the calculation of the simple
                    moving average
                wma_period: int Controls the period of the weighted movement indicator calculation
                ema_period: int Controls the period of the exponential moving average indicator calculation
                momentum_period: int Controls the period of the momentum indicator calculation.
                rate_of_change_period: int Controls the period of the rate of change calculation
                oscillator_period: int Controls the period of the stochastic oscillator calculation
                cci_period: int Controls the period of the commodity channel index calculation

        Returns:
            Dictionary mapping the string representation of stock tickers to their respective generated
                numpy.ndarray data block.
            Each data block will have n columns where n <= data_block_length, and all values in the
                data block will be either -1 or 1 as per the definition of trend deterministic data
                found in (2014, Pael, Shah, et al)
        """
        return self._generate_agnostic_data(*args, **kwargs)

    def generate_prediction_data(self, *args, **kwargs):
        kwargs['predict'] = True
        return self._generate_agnostic_data(*args, **kwargs)

    def load_configuration(self, parser: "configparser.ConfigParser"):
        section = config_util.create_type_section(parser, self)
        if not parser.has_option(section.name, _ENABLED_CONFIG_ID):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, _ENABLED_CONFIG_ID)
        if enabled:
            data_provider_registry.registry.register_provider(
                data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID, self)

    def write_default_configuration(self, section: "configparser.SectionProxy"):
        section[_ENABLED_CONFIG_ID] = 'True'

    def __init__(self):

        super(TrendDeterministicBlockProvider, self).__init__()
        configurable_registry.config_registry.register_configurable(self)
        self.default_kwargs = {
            "sma_period": 10,
            "wma_period": 10,
            "ema_period": [22],
            "macd_ema_1_period": 26,
            "macd_ema_2_period": 12,
            "oscillator_period": 10,
            "momentum_period": 10,
            "rate_of_change_period": 10,
            'cci_period': 10,
            "trend_strength_labelling": False,
            "trend_lookahead": 1
        }


provider = TrendDeterministicBlockProvider()
