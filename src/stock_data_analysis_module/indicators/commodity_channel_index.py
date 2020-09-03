from typing import Union, List
import numpy as np


def commodity_channel_index(
                    closing_data: Union["np.ndarray", List[float]],
                    high_data: Union["np.ndarray", List[float]],
                    low_data: Union["np.ndarray", List[float]],
                    period: int
                    ) -> np.ndarray:
    if len(closing_data) < period:
        raise ValueError("Unable to calculate commodity channel index for data of length %d"
                         "and period of %d" % (len(closing_data), period))
    ret_indicator = np.zeros((len(closing_data)-period)+1)
    typical_price = np.zeros(len(closing_data))
    for i in range(len(typical_price)):
        typical_price[i] = (high_data[i] + closing_data[i] + low_data[i]) / 3
    for i in range(len(ret_indicator)):
        price_window = typical_price[i:i+period]
        average_price = np.average(price_window)
        mean_deviation = 0
        for j in range(len(price_window)):
            mean_deviation += abs(price_window[j] - average_price)
        mean_deviation /= period
        if mean_deviation == 0:
            ret_indicator[i] = 0
        else:
            ret_indicator[i] = (price_window[-1] - average_price) / (.015 * mean_deviation)
    return ret_indicator
