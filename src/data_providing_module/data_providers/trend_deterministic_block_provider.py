"""Data Provider Implementation Module for constructing data based on Trend Deterministic Data

Guiding Principles of this implementation are available in the paper
"Predicting stock and stock price index movement using Trend Deterministic Data
Preparation and machine learning techniques" (2014, Pael, Shah, et al)

As with other Data Provider implementations, this provider is not intended to be
instantiated outside of this module. All interactions with this provider are to
be done through the global DataProviderRegistry's registration and data passing systems.

"""

import configparser
import datetime

import numpy as np

from data_providing_module.data_providers import data_provider_static_names
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


class TrendDeterministicBlockProvider(data_provider_registry.DataProviderBase):

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
        if not args:
            raise ValueError("Expected %d positional argument but received %d" % (1, len(args)))
        data_block_length = args[0]
        max_additional_period = 0
        for key, value in self.default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = self.default_kwargs[key]

            if key.endswith("period") and value > max_additional_period:
                max_additional_period = value
        # as most indicators are translated into trend deterministic data by comparison with the
        # previous period's result, and additional period of trading data is required.
        max_additional_period += 1

        padded_data_block_length = max_additional_period + data_block_length
        end_date = datetime.datetime.now().isoformat()[:10].replace('-', '/')
        data_retriever = period_data_retriever.PeriodDataRetriever(
            [
                stock_data_table.HIGH_PRICE_COLUMN_NAME,
                stock_data_table.LOW_PRICE_COLUMN_NAME,
                stock_data_table.CLOSING_PRICE_COLUMN_NAME,
            ],
            end_date
        )
        ret_blocks = {}
        for ticker, _ in data_retriever.data_sources.items():
            ticker_data = data_retriever.retrieve_data(ticker, max_rows=padded_data_block_length)
            ticker_data = np.array(ticker_data, dtype=np.float32)
            high = ticker_data[:, 0]
            low = ticker_data[:, 1]
            close = ticker_data[:, 2]
            if len(high) < max_additional_period:
                len_warning = (
                        "Could not process %s into an indicator block, "
                        "needed %d days of trading data but received %d" %
                        (ticker, max_additional_period, len(high))
                )
                logger.logger.log(logger.WARNING, len_warning)
                continue

            actual_trends = [1 if close[i] <= close[i+1] else -1 for i in range(len(close)-1)]

            sma = moving_average.SMA(close, kwargs['sma_period'])
            for i in range(len(sma)-1):
                sma[i] = 1 if sma[i] <= sma[i+1] else -1
            sma = sma[-data_block_length-1:-1]

            wma = moving_average.WMA(close, kwargs['wma_period'])
            for i in range(len(wma)-1):
                wma[i] = 1 if wma[i] <= wma[i+1] else -1
            wma = wma[-data_block_length-1:-1]

            ema = moving_average.EMA(close, kwargs['ema_period'])
            for i in range(len(ema)-1):
                ema[i] = 1 if ema[i] <= ema[i+1] else -1
            ema = ema[-data_block_length-1:-1]

            macd_ema_1 = moving_average.EMA(close, kwargs['macd_ema_1_period'])
            macd_ema_2 = moving_average.EMA(close, kwargs['macd_ema_2_period'])
            macd_ema_2 = macd_ema_2[-len(macd_ema_1):]
            macd = macd_ema_2 - macd_ema_1
            for i in range(len(macd)-1):
                macd[i] = 1 if macd[i] <= macd[i+1] else -1
            macd = macd[-data_block_length-1:-1]

            price_momentum = momentum.momentum(close, kwargs['momentum_period'])
            for i in range(len(price_momentum)-1):
                price_momentum[i] = 1 if price_momentum[i] <= price_momentum[i+1] else -1
            price_momentum = price_momentum[-data_block_length-1:-1]

            roc = rate_of_change.rate_of_change(close, kwargs['rate_of_change_period'])
            for i in range(len(roc)-1):
                roc[i] = 1 if roc[i] <= roc[i+1] else -1
            roc = roc[-data_block_length-1:-1]

            oscillator = stochastic_oscillator.stochastic_oscillator(close, high,
                                                                     low, kwargs['oscillator_period'])
            for i in range(len(oscillator)-1):
                if oscillator[i+1] >= 80:
                    oscillator[i] = -1
                elif oscillator[i+1] <= 20:
                    oscillator[i] = 1
                else:
                    oscillator[i] = 1 if oscillator[i] <= oscillator[i+1] else -1
            oscillator = oscillator[-data_block_length-1:-1]

            cci = commodity_channel_index.commodity_channel_index(close, high,
                                                                  low, kwargs['cci_period'])

            for i in range(len(cci)-1):
                if cci[i+1] >= 200:
                    cci[i] = -1
                elif cci[i+1] <= -200:
                    cci[i] = 1
                else:
                    cci[i] = 1 if cci[i] <= cci[i+1] else -1
            cci = cci[-data_block_length-1:-1]

            ad_oscillator = stochastic_oscillator.ad_oscillator(close, high, low)
            for i in range(len(ad_oscillator)-1):
                ad_oscillator[i] = 1 if ad_oscillator[i] <= ad_oscillator[i+1] else -1
            ad_oscillator = ad_oscillator[-data_block_length-1:-1]

            stock_data_block = [sma, wma, ema.flatten(), macd.flatten(),
                                price_momentum, roc, oscillator,
                                ad_oscillator, cci]
            min_len = len(sma)
            for data in stock_data_block:
                if min_len > len(data):
                    min_len = len(data)
            for i in range(len(stock_data_block)):
                stock_data_block[i] = stock_data_block[i][-min_len:]

            stock_data_block = np.array(stock_data_block)
            actual_trends = actual_trends[-min_len:]
            ret_blocks[ticker] = (stock_data_block[:, :-1], np.array(actual_trends[1:]))
        return ret_blocks

    def generate_prediction_data(self, *args, **kwargs):
        if not args:
            raise ValueError("Expected %d positional argument but received %d" % (1, len(args)))
        data_block_length = args[0]
        max_additional_period = 0
        for key, value in self.default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = self.default_kwargs[key]

            if key.endswith("period") and value > max_additional_period:
                max_additional_period = value
        # as most indicators are translated into trend deterministic data by comparison with the
        # previous period's result, and additional period of trading data is required.
        max_additional_period += 1

        padded_data_block_length = max_additional_period + data_block_length
        end_date = datetime.datetime.now().isoformat()[:10].replace('-', '/')
        data_retriever = period_data_retriever.PeriodDataRetriever(
            [
                stock_data_table.HIGH_PRICE_COLUMN_NAME,
                stock_data_table.LOW_PRICE_COLUMN_NAME,
                stock_data_table.CLOSING_PRICE_COLUMN_NAME,
            ],
            end_date
        )
        ret_blocks = {}
        for ticker, _ in data_retriever.data_sources.items():
            ticker_data = data_retriever.retrieve_data(ticker, max_rows=padded_data_block_length)
            ticker_data = np.array(ticker_data, dtype=np.float32)
            high = ticker_data[:, 0]
            low = ticker_data[:, 1]
            close = ticker_data[:, 2]
            if len(high) < max_additional_period:
                len_warning = (
                        "Could not process %s into an indicator block, "
                        "needed %d days of trading data but received %d" %
                        (ticker, max_additional_period, len(high))
                )
                logger.logger.log(logger.WARNING, len_warning)
                continue

            actual_trends = [1 if close[i] <= close[i + 1] else -1 for i in range(len(close) - 1)]

            sma = moving_average.SMA(close, kwargs['sma_period'])
            for i in range(len(sma) - 1):
                sma[i] = 1 if sma[i] <= sma[i + 1] else -1
            sma = sma[-data_block_length - 1:-1]

            wma = moving_average.WMA(close, kwargs['wma_period'])
            for i in range(len(wma) - 1):
                wma[i] = 1 if wma[i] <= wma[i + 1] else -1
            wma = wma[-data_block_length - 1:-1]

            ema = moving_average.EMA(close, kwargs['ema_period'])
            for i in range(len(ema) - 1):
                ema[i] = 1 if ema[i] <= ema[i + 1] else -1
            ema = ema[-data_block_length - 1:-1]

            macd_ema_1 = moving_average.EMA(close, kwargs['macd_ema_1_period'])
            macd_ema_2 = moving_average.EMA(close, kwargs['macd_ema_2_period'])
            macd_ema_2 = macd_ema_2[-len(macd_ema_1):]
            macd = macd_ema_2 - macd_ema_1
            for i in range(len(macd) - 1):
                macd[i] = 1 if macd[i] <= macd[i + 1] else -1
            macd = macd[-data_block_length - 1:-1]

            price_momentum = momentum.momentum(close, kwargs['momentum_period'])
            for i in range(len(price_momentum) - 1):
                price_momentum[i] = 1 if price_momentum[i] <= price_momentum[i + 1] else -1
            price_momentum = price_momentum[-data_block_length - 1:-1]

            roc = rate_of_change.rate_of_change(close, kwargs['rate_of_change_period'])
            for i in range(len(roc) - 1):
                roc[i] = 1 if roc[i] <= roc[i + 1] else -1
            roc = roc[-data_block_length - 1:-1]

            oscillator = stochastic_oscillator.stochastic_oscillator(close, high,
                                                                     low, kwargs['oscillator_period'])
            for i in range(len(oscillator) - 1):
                if oscillator[i + 1] >= 80:
                    oscillator[i] = -1
                elif oscillator[i + 1] <= 20:
                    oscillator[i] = 1
                else:
                    oscillator[i] = 1 if oscillator[i] <= oscillator[i + 1] else -1
            oscillator = oscillator[-data_block_length - 1:-1]

            cci = commodity_channel_index.commodity_channel_index(close, high,
                                                                  low, kwargs['cci_period'])

            for i in range(len(cci) - 1):
                if cci[i + 1] >= 200:
                    cci[i] = -1
                elif cci[i + 1] <= -200:
                    cci[i] = 1
                else:
                    cci[i] = 1 if cci[i] <= cci[i + 1] else -1
            cci = cci[-data_block_length - 1:-1]

            ad_oscillator = stochastic_oscillator.ad_oscillator(close, high, low)
            for i in range(len(ad_oscillator) - 1):
                ad_oscillator[i] = 1 if ad_oscillator[i] <= ad_oscillator[i + 1] else -1
            ad_oscillator = ad_oscillator[-data_block_length - 1:-1]

            stock_data_block = [sma, wma, ema.flatten(), macd.flatten(),
                                price_momentum, roc, oscillator,
                                ad_oscillator, cci]
            min_len = len(sma)
            for data in stock_data_block:
                if min_len > len(data):
                    min_len = len(data)
            for i in range(len(stock_data_block)):
                stock_data_block[i] = stock_data_block[i][-min_len:]

            stock_data_block = np.array(stock_data_block)
            actual_trends = actual_trends[-min_len:]
            # the only difference here is that the full data block should be returned (as
            # this is what should be used to predict the next day's state.
            # However, the actual trends that were calculated here are returned as well
            # so that a baseline accuracy can be given for the prediction
            # kind of like a "hey this model claims 80% accuracy but it's only been 50% accurate on
            # this trial set. That's not reliable then"
            ret_blocks[ticker] = (stock_data_block, np.array(actual_trends[1:]))
        return ret_blocks

    def load_configuration(self, parser: "configparser.ConfigParser"):
        section = config_util.create_type_section(parser, self)
        if not parser.has_option(section.name, _ENABLED_CONFIG_ID):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, _ENABLED_CONFIG_ID)
        if not enabled:
            data_provider_registry.registry.deregister_provider(
                data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID)

    def write_default_configuration(self, section: "configparser.SectionProxy"):
        section[_ENABLED_CONFIG_ID] = 'True'

    def __init__(self):

        super(TrendDeterministicBlockProvider, self).__init__()
        data_provider_registry.registry.register_provider(
            data_provider_static_names.TREND_DETERMINISTIC_BLOCK_PROVIDER_ID, self)
        self.default_kwargs = {
            "sma_period": 10,
            "wma_period": 10,
            "ema_period": 22,
            "macd_ema_1_period": 26,
            "macd_ema_2_period": 12,
            "oscillator_period": 10,
            "momentum_period": 10,
            "rate_of_change_period": 10,
            'cci_period': 10
        }


provider = TrendDeterministicBlockProvider()
