"""Data Provider Implementation Module for constructing data for use in Multi-stage Stock Prediction

Guiding Principles of this implementation are available in the paper
"Predicting Stock Market Index using Fusion of Machine Learning Techniques" (Patel et. al.)

As with other Data Provider implementations, this provider is not intended to be
instantiated outside of this module. All interactions with this provider are to
be done through the global DataProviderRegistry's registration and data passing systems.
"""

import configparser
import datetime
from typing import List, Dict, Tuple

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


class ClosingPriceRegressionBlockProvider(data_provider_registry.DataProviderBase):

    def __init__(self):
        super(ClosingPriceRegressionBlockProvider, self).__init__()
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
            "lookahead_factor": 2
        }
        configurable_registry.config_registry.register_configurable(self)

    def _generate_agnostic_data(self, *args, **kwargs):
        if not args:
            raise ValueError("Expected 1 positional argument but received 0")

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
        ret_blocks: Dict[Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]]] = {}

        for ticker, _ in data_retriever.data_sources.items():
            ticker_data = data_retriever.retrieve_data(ticker, max_rows=padded_data_block_length)
            ticker_data = np.array(ticker_data, dtype=np.float32)
            high = ticker_data[:, 0]
            high = np.array(list(reversed(high)))
            low = ticker_data[:, 1]
            low = np.array(list(reversed(low)))
            close = ticker_data[:, 2]
            close = np.array(list(reversed(close)))
            if len(high) < max_additional_period:
                len_warning = (
                        "Could not process %s into an indicator block, "
                        "needed %d days of trading data but received %d" %
                        (ticker, max_additional_period, len(high))
                )
                logger.logger.log(logger.WARNING, len_warning)
                continue

            lookahead = kwargs['lookahead_factor']

            sma = moving_average.SMA(close, kwargs['sma_period'])

            wma = moving_average.WMA(close, kwargs['wma_period'])

            emas = []
            for period in kwargs['ema_period']:
                ema = moving_average.EMA(close, period)
                emas.append(ema.flatten())

            macd_ema_1 = moving_average.EMA(close, kwargs['macd_ema_1_period'])
            macd_ema_2 = moving_average.EMA(close, kwargs['macd_ema_2_period'])
            macd_ema_2 = macd_ema_2[-len(macd_ema_1):]
            macd = macd_ema_2 - macd_ema_1

            price_momentum = momentum.momentum(close, kwargs['momentum_period'])

            roc = rate_of_change.rate_of_change(close, kwargs['rate_of_change_period'])

            oscillator = stochastic_oscillator.stochastic_oscillator(close, high,
                                                                     low, kwargs['oscillator_period'])

            cci = commodity_channel_index.commodity_channel_index(close, high,
                                                                  low, kwargs['cci_period'])

            ad_oscillator = stochastic_oscillator.ad_oscillator(close, high, low)

            # the idea is that once these indicators are in a block that is all one length,
            # the earliest period viable for prediction is max_additional_period. The
            # closing price to be predicted by the samples for that period would then be
            # max_additional_period + kwargs['lookahead_factor']. Looking ahead then is
            # just a matter of iterating through the periods and if the sum above is not
            # out of range for the closing price, then it is added to the training set.
            stock_data_block = [sma, wma]
            stock_data_block.extend(emas)
            stock_data_block.extend([macd.flatten(), price_momentum, roc, oscillator, ad_oscillator, cci])
            min_len = len(sma)
            for data in stock_data_block:
                if min_len > len(data):
                    min_len = len(data)
            for i in range(len(stock_data_block)):
                stock_data_block[i] = stock_data_block[i][-min_len:]

            stock_data_block = np.array(stock_data_block)
            closing_prices = []
            for i in range(stock_data_block.shape[1]):
                calc_index = i + max_additional_period + kwargs['lookahead_factor']
                if calc_index < len(close):
                    closing_prices.append(close[calc_index])
                else:
                    break
            if 'predict' in kwargs:
                ret_blocks[ticker] = (stock_data_block[:, -data_block_length:len(closing_prices)+1],
                                      closing_prices[-data_block_length:])
            else:
                ret_blocks[ticker] = (stock_data_block[:, -data_block_length:len(closing_prices)],
                                      closing_prices[-data_block_length:])
        return ret_blocks

    def generate_data(self, *args, **kwargs):
        return self._generate_agnostic_data(*args, **kwargs)

    def generate_prediction_data(self, *args, **kwargs):
        kwargs['predict'] = True
        return self._generate_agnostic_data(*args, **kwargs)

    def load_configuration(self, parser: "ConfigParser"):
        section = config_util.create_type_section(parser, self)
        if not parser.has_option(section.name, _ENABLED_CONFIG_ID):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, _ENABLED_CONFIG_ID)
        if enabled:
            data_provider_registry.registry.register_provider(
                data_provider_static_names.CLOSING_PRICE_REGRESSION_BLOCK_PROVIDER_ID,
                self
            )

    def write_default_configuration(self, section: "SectionProxy"):
        section[_ENABLED_CONFIG_ID] = 'True'


provider = ClosingPriceRegressionBlockProvider()
