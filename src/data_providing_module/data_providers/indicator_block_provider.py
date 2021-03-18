"""Data Provider implementation module for constructing data based on standard stock indicators

The data provider in this module is not indented to be instantiated outside of this module. Instead, upon the importing
of this module, the provider will create an instance of itself and register itself with the global DataProviderRegistry.
After this, data consumers can register themselves as recipients of data from this provider using the id located at
data_provider_static_names.INDICATOR_BLOCK_PROVIDER_ID.

Detailed argument list that can be provided to this provider can be found in the generate_data method.

TODO[Colton Freitas] Extract repeated code from generate_prediction_data and generate_data after global style rewrite
"""

import configparser
import datetime

import numpy

from data_providing_module import configurable_registry
from data_providing_module import data_provider_registry
from data_providing_module.data_providers import data_provider_static_names

from general_utils.config import config_util
from general_utils.logging import logger
from general_utils.mysql_management.mysql_tables import stock_data_table

from stock_data_analysis_module.data_processing_module.data_retrieval_module import ranged_data_retriever
from stock_data_analysis_module.indicators import moving_average
from stock_data_analysis_module.indicators import bollinger_band
from stock_data_analysis_module.indicators import stochastic_oscillator

_ENABLED_CONFIG_ID = "enabled"


def _standardize_price_data(price_data):
    ret_data = numpy.copy(price_data)
    ret_data = ret_data.flatten()
    max_price = numpy.max(ret_data)
    min_price = numpy.min(ret_data)
    for i in range(len(ret_data)):
        ret_data[i] = (ret_data[i]-min_price)/max_price
    return ret_data.reshape(price_data.shape)


class IndicatorBlockProvider(data_provider_registry.DataProviderBase):
    """Data Provider that will provide data constructed using stock indicators normally used by stock traders

    Details on these indicators can be found in the modules of the indicators package.

    Additionally, this provider provides support for configurable parameters through the configuration file. These
    parameters are listed in the Configurable Parameters section.

    Configurable Parameters:
        enable: Whether this provider is enabled for consumers to receive data from.
    """

    def generate_prediction_data(self, *args, **kwargs):
        """Generates data for a Consumer wanting to make predictions about the next day's state.

        This method is identical to generate_data for all but the return values. As such, for arguments
            and further details, see generate_data.

        Returns:
            List[Tuple[str, numpy.ndarray, float, float]]. Broken down, for every stock, there is a tuple
                containing the ticker, the data block generated, the average price, and the average volume.
                The average price and volume is to allow for the original magnitudes of the prices and volumes to
                be reconstructed should the predictions require it.
            For a breakdown of the rows in the data block, see generate_data's documentation in the Returns section.
        """
        if len(args) < 1:
            raise ValueError("Expected %d positional argument but received %d" % (1, len(args)))
        data_block_length = args[0]
        max_additional_period = 0
        for key, value in self.default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = self.default_kwargs[key]

            if key.endswith("period") and value > max_additional_period:
                max_additional_period = value
        padded_data_block_length = max_additional_period + data_block_length
        start_date = datetime.datetime.now() - datetime.timedelta(weeks=(padded_data_block_length + 360) // 5)
        start_date = start_date.isoformat()[:10].replace('-', '/')
        end_date = datetime.datetime.now().isoformat()[:10].replace('-', '/')
        data_retriever = ranged_data_retriever.RangedDataRetriever(
            [
                stock_data_table.HIGH_PRICE_COLUMN_NAME,
                stock_data_table.LOW_PRICE_COLUMN_NAME,
                stock_data_table.CLOSING_PRICE_COLUMN_NAME,
                stock_data_table.VOLUME_COLUMN_NAME
            ],
            start_date,
            end_date)
        ret_blocks = []
        for ticker, sources in data_retriever.data_sources.items():
            ticker_data = data_retriever.retrieveData(ticker, sources[0])
            ticker_data = numpy.array(ticker_data, dtype=numpy.float32)
            high = ticker_data[:, 0]
            low = ticker_data[:, 1]
            close = ticker_data[:, 2]
            volume = ticker_data[:, 3]

            # high, low, close, volume = ticker_data  # unpack manually
            avg_high = numpy.average(high)
            avg_low = numpy.average(low)
            avg_close = numpy.average(close)
            avg_price = ((avg_high * len(high)) + (avg_low * len(high)) + (avg_close * len(high))) / (len(high) * 3)
            avg_vol = numpy.average(volume)
            std_high = [(high[i] - avg_price) / avg_price
                        for i in range(len(high))]
            std_low = [(low[i] - avg_price) / avg_price
                       for i in range(len(high))]
            std_close = [(close[i] - avg_price) / avg_price
                         for i in range(len(high))]
            volume = [(volume[i] - avg_vol) / avg_vol
                      for i in range(len(volume))]
            if len(std_high) < padded_data_block_length:
                len_warning = (
                        "Could not process %s into an indicator block, "
                        "needed %d days of trading data but received %d" %
                        (ticker, padded_data_block_length, len(std_high))
                )
                logger.logger.log(logger.WARNING, len_warning)
                continue
            sma = moving_average.SMA(std_close, kwargs['sma_period'])
            sma = sma[-data_block_length:]
            boll_band = bollinger_band.bollinger_band(std_high, std_low, std_close,
                                                      smoothing_period=kwargs["bollinger_band_period"],
                                                      standard_deviations=kwargs["bollinger_band_stdev"]
                                                      )
            oscillator = stochastic_oscillator.stochastic_oscillator(close, high,
                                                                     low, kwargs['oscillator_period'])
            oscillator = oscillator[-data_block_length:]

            oscillator /= 100
            data_block = numpy.zeros((8, data_block_length), dtype=numpy.float32)
            data_block[0] = std_high[-data_block_length:]
            data_block[1] = std_low[-data_block_length:]
            data_block[2] = std_close[-data_block_length:]
            data_block[3] = volume[-data_block_length:]
            data_block[4] = sma
            data_block[5] = boll_band[0][-data_block_length:]
            data_block[6] = boll_band[1][-data_block_length:]
            data_block[7] = oscillator
            ret_blocks.append((ticker, data_block, avg_price, avg_vol))
        return ret_blocks

    def write_default_configuration(self, section: "configparser.SectionProxy"):
        """Writes default configuration values into the SectionProxy provided.

        For more details see abstract class documentation.
        """
        section[_ENABLED_CONFIG_ID] = "True"

    def load_configuration(self, parser: "configparser.ConfigParser"):
        """Attempts to load the configurable parameters for this provider from the provided parser.

        For more details see abstract class documentation.
        """
        section = config_util.create_type_section(parser, self)
        if not parser.has_option(section.name, _ENABLED_CONFIG_ID):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, _ENABLED_CONFIG_ID)
        if enabled:
            data_provider_registry.registry.register_provider(
                data_provider_static_names.INDICATOR_BLOCK_PROVIDER_ID, self)

    def generate_data(self, *args, **kwargs):
        """Generates data using stock indicators over a set period of time

        Generates blocks (numpy arrays) of data using indicators that are used by normal stock traders.
        These include bollinger bands, simple moving average and the stochastic oscillator.

        The types of data that get fed into these algorithms come from the high, low, closing, and volume columns
        of the data tables in the database. Additionally, these values are standardized to allow algorithms to draw
        conclusions based off the relative change in the stock, and not be blinded by the magnitude of the prices or
        volumes.

        This standardization process is performed by calculating the average price across the highs, lows, and closing
        prices of the stock, then every element in each of the lists is updated according to the following equation
        (assume that price is the high, low, or closing price being modified):
            (price - avg_price) / avg_price
        The same process is also performed on the volume data.

        Additionally, consumers are required to pass in a positional argument through *args, and may pass in
        keyword arguments. These are covered in the Arguments section below

        Arguments:
            *args:
                Only one positional argument is required.
                data_block_length: int This controls how many columns will
                    be present in the return data block. As a note the data block will always have 8 rows.
            **kwargs:
                Several keyword arguments are supported.
                sma_period: int Controls how many days are considered in the calculation of the simple moving average.
                    For a given day x, the previous x-sma_period days will be used
                bollinger_band_stdev: int Controls how many standard deviations will be used in the calculation
                    of the bollinger bands
                bollinger_band_period: int Controls how many days will be used in the calculation of the bollinger
                    bands.
                oscillator_period: int Controls the number of days used in the calculation of the stochastic oscillator

        Returns:
            Numpy.ndarray object with three dimensions. This is effectively a 3D matrix of data blocks, where each
                data block will have 8 rows and data_block_length columns.
            Each data block row corresponds to one data type or calculated indicator values, are listed below:
                0: high price
                1: low price
                2: closing price
                3: volume
                4: simple moving average (SMA)
                5: upper bollinger band
                6: lower bollinger band
                7: stochastic oscillator
        """
        if len(args) < 1:
            raise ValueError("Expected %d positional argument but received %d" % (1, len(args)))
        data_block_length = args[0]
        max_additional_period = 0
        for key, value in self.default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = self.default_kwargs[key]

            if key.endswith("period") and value > max_additional_period:
                max_additional_period = value
        padded_data_block_length = max_additional_period + data_block_length
        start_date = datetime.datetime.now() - datetime.timedelta(weeks=(padded_data_block_length + 360) // 5)
        start_date = start_date.isoformat()[:10].replace('-', '/')
        end_date = datetime.datetime.now().isoformat()[:10].replace('-', '/')
        data_retriever = ranged_data_retriever.RangedDataRetriever(
            [
                stock_data_table.HIGH_PRICE_COLUMN_NAME,
                stock_data_table.LOW_PRICE_COLUMN_NAME,
                stock_data_table.CLOSING_PRICE_COLUMN_NAME,
                stock_data_table.VOLUME_COLUMN_NAME
            ],
            start_date,
            end_date)
        ret_blocks = []
        for ticker, sources in data_retriever.data_sources.items():
            ticker_data = data_retriever.retrieveData(ticker, sources[0])
            ticker_data = numpy.array(ticker_data, dtype=numpy.float32)
            high = ticker_data[:, 0]
            low = ticker_data[:, 1]
            close = ticker_data[:, 2]
            volume = ticker_data[:, 3]

            # high, low, close, volume = ticker_data  # unpack manually

            std_high = _standardize_price_data(high)
            std_close = _standardize_price_data(close)
            std_low = _standardize_price_data(low)
            volume = _standardize_price_data(volume)

            if len(std_high) < padded_data_block_length:
                len_warning = (
                        "Could not process %s into an indicator block, "
                        "needed %d days of trading data but received %d" %
                        (ticker, padded_data_block_length, len(std_high))
                )
                logger.logger.log(logger.WARNING, len_warning)
                continue
            sma = moving_average.SMA(std_close, kwargs['sma_period'])
            sma = sma[-data_block_length:]
            boll_band = bollinger_band.bollinger_band(std_high, std_low, std_close,
                                                      smoothing_period=kwargs["bollinger_band_period"],
                                                      standard_deviations=kwargs["bollinger_band_stdev"]
                                                      )
            oscillator = stochastic_oscillator.stochastic_oscillator(close, high,
                                                                     low, kwargs['oscillator_period'])
            oscillator = oscillator[-data_block_length:]

            oscillator /= 100
            data_block = numpy.zeros((8, data_block_length), dtype=numpy.float32)
            data_block[0] = std_high[-data_block_length:]
            data_block[1] = std_low[-data_block_length:]
            data_block[2] = std_close[-data_block_length:]
            data_block[3] = volume[-data_block_length:]
            data_block[4] = sma
            data_block[5] = boll_band[0][-data_block_length:]
            data_block[6] = boll_band[1][-data_block_length:]
            data_block[7] = oscillator
            ret_blocks.append(data_block)
        return numpy.array(ret_blocks, dtype=numpy.float32)

    def __init__(self):
        """Initializes IndicatorBlockProvider and registers the instance with the global DataProviderRegistry

        """

        super(IndicatorBlockProvider, self).__init__()
        configurable_registry.config_registry.register_configurable(self)
        self.default_kwargs = {
            "sma_period": 50,
            "bollinger_band_stdev": 2,
            "bollinger_band_period": 20,
            "oscillator_period": 17
        }


provider = IndicatorBlockProvider()
