from configparser import ConfigParser, SectionProxy

from DataProvidingModule.DataProviderRegistry import registry, DataProviderBase
from datetime import datetime as dt, timedelta as td
from StockDataAnalysisModule.DataProcessingModule.DataRetrievalModule.RangedDataRetriever import RangedDataRetriever
from StockDataAnalysisModule.Indicators.MovingAverage import SMA
from StockDataAnalysisModule.Indicators.BollingerBand import bollinger_band
from StockDataAnalysisModule.Indicators.StochasticOscillator import stochastic_oscillator
from GeneralUtils.Config import ConfigUtil as cfgUtil
import numpy as np


class IndicatorBlockProvider(DataProviderBase):

    def generatePredictionData(self, login_credentials, *args, **kwargs):
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
        start_date = dt.now() - td(weeks=(padded_data_block_length + 360) // 5)
        start_date = start_date.isoformat()[:10].replace('-', '/')
        end_date = dt.now().isoformat()[:10].replace('-', '/')
        data_retriever = RangedDataRetriever(login_credentials, ['high_price, low_price, close_price, volume_data'],
                                             start_date, end_date)
        ret_blocks = []
        for ticker, sources in data_retriever.data_sources.items():
            ticker_data = data_retriever.retrieveData(ticker, sources[0])
            ticker_data = np.array(ticker_data, dtype=np.float32)
            high = ticker_data[:, 0]
            low = ticker_data[:, 1]
            close = ticker_data[:, 2]
            volume = ticker_data[:, 3]

            # high, low, close, volume = ticker_data  # unpack manually
            avg_high = np.average(high)
            avg_low = np.average(low)
            avg_close = np.average(close)
            avg_price = ((avg_high * len(high)) + (avg_low * len(high)) + (avg_close * len(high))) / (len(high) * 3)
            avg_vol = np.average(volume)
            std_high = [(high[i] - avg_price) / avg_price
                        for i in range(len(high))]
            std_low = [(low[i] - avg_price) / avg_price
                       for i in range(len(high))]
            std_close = [(close[i] - avg_price) / avg_price
                         for i in range(len(high))]
            volume = [(volume[i] - avg_vol) / avg_vol
                      for i in range(len(volume))]
            if len(std_high) < padded_data_block_length:
                print("Could not process %s into an indicator block, needed %d days of trading data but received %d" %
                      (ticker, padded_data_block_length, len(std_high)))
                continue
            sma = SMA(std_close, kwargs['sma_period'])
            sma = sma[-data_block_length:]
            boll_band = bollinger_band(std_high, std_low, std_close,
                                       smoothing_period=kwargs["bollinger_band_period"],
                                       standard_deviations=kwargs["bollinger_band_stdev"]
                                       )
            oscillator = stochastic_oscillator(close, high,
                                               low, kwargs['oscillator_period'])
            oscillator = oscillator[-data_block_length:]

            oscillator /= 100
            data_block = np.zeros((8, data_block_length), dtype=np.float32)
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

    def write_default_configuration(self, section: "SectionProxy"):
        section["enabled"] = "True"

    def load_configuration(self, parser: "ConfigParser"):
        section = cfgUtil.create_type_section(parser, self)
        if not parser.has_option(section.name, "enabled"):
            self.write_default_configuration(section)
        enabled = parser.getboolean(section.name, "enabled")
        if not enabled:
            registry.deregisterProvider("IndicatorBlockProvider")

    def generateData(self, login_credentials, *args, **kwargs):
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
        start_date = dt.now() - td(weeks=(padded_data_block_length+360)//5)
        start_date = start_date.isoformat()[:10].replace('-', '/')
        end_date = dt.now().isoformat()[:10].replace('-', '/')
        data_retriever = RangedDataRetriever(login_credentials, ['high_price, low_price, close_price, volume_data'],
                                             start_date, end_date)
        ret_blocks = []
        for ticker, sources in data_retriever.data_sources.items():
            ticker_data = data_retriever.retrieveData(ticker, sources[0])
            ticker_data = np.array(ticker_data, dtype=np.float32)
            high = ticker_data[:, 0]
            low = ticker_data[:, 1]
            close = ticker_data[:, 2]
            volume = ticker_data[:, 3]

            # high, low, close, volume = ticker_data  # unpack manually
            avg_high = np.average(high)
            avg_low = np.average(low)
            avg_close = np.average(close)
            avg_price = ((avg_high * len(high)) + (avg_low * len(high)) + (avg_close * len(high))) / (len(high) * 3)
            avg_vol = np.average(volume)
            std_high = [(high[i] - avg_price) / avg_price
                    for i in range(len(high))]
            std_low = [(low[i] - avg_price) / avg_price
                    for i in range(len(high))]
            std_close = [(close[i] - avg_price) / avg_price
                    for i in range(len(high))]
            volume = [(volume[i] - avg_vol) / avg_vol
                    for i in range(len(volume))]
            if len(std_high) < padded_data_block_length:
                print("Could not process %s into an indicator block, needed %d days of trading data but received %d" %
                      (ticker, padded_data_block_length, len(std_high)))
                continue
            sma = SMA(std_close, kwargs['sma_period'])
            sma = sma[-data_block_length:]
            boll_band = bollinger_band(std_high, std_low, std_close,
                                       smoothing_period=kwargs["bollinger_band_period"],
                                       standard_deviations=kwargs["bollinger_band_stdev"]
                                       )
            oscillator = stochastic_oscillator(close, high,
                                               low, kwargs['oscillator_period'])
            oscillator = oscillator[-data_block_length:]

            oscillator /= 100
            data_block = np.zeros((8, data_block_length), dtype=np.float32)
            data_block[0] = std_high[-data_block_length:]
            data_block[1] = std_low[-data_block_length:]
            data_block[2] = std_close[-data_block_length:]
            data_block[3] = volume[-data_block_length:]
            data_block[4] = sma
            data_block[5] = boll_band[0][-data_block_length:]
            data_block[6] = boll_band[1][-data_block_length:]
            data_block[7] = oscillator
            ret_blocks.append(data_block)
        return np.array(ret_blocks, dtype=np.float32)

    def __init__(self):
        super(IndicatorBlockProvider, self).__init__()
        registry.registerProvider("IndicatorBlockProvider", self)
        self.default_kwargs = {
            "sma_period": 50,
            "bollinger_band_stdev": 2,
            "bollinger_band_period": 20,
            "oscillator_period": 17
        }


try:
    provider = provider
except NameError:
    provider = IndicatorBlockProvider()