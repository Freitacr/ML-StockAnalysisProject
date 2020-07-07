from typing import Union, List, Tuple
import numpy as np
from StockDataAnalysisModule.Indicators.MovingAverage import SMA
from statistics import stdev


def bollinger_band(
                    high_data: Union["np.ndarray", List[float]],
                    low_data: Union["np.ndarray", List[float]],
                    close_data: Union["np.ndarray", List[float]],
                    smoothing_period: int = 20,
                    standard_deviations: int = 2
                 ) -> Tuple["np.ndarray", "np.ndarray"]:
    typical_price = (np.array(high_data) + np.array(low_data) + np.array(close_data)) / 3
    if len(typical_price) < smoothing_period:
        raise ValueError("Unable to calculate bollinger bands for data of length %d and a smoothing period of %d" %
                         (len(typical_price), smoothing_period))
    ret_size = len(typical_price) - smoothing_period
    moving_average = SMA(typical_price, smoothing_period)
    standard_dev = [stdev(typical_price[i:i+smoothing_period], moving_average[i]) for i in range(len(moving_average))]
    standard_dev = np.array(standard_dev)
    standard_dev *= standard_deviations
    ret_upper_band = moving_average + standard_dev
    ret_lower_band = moving_average - standard_dev
    return ret_upper_band, ret_lower_band
