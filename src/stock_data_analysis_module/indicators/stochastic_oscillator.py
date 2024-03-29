from typing import Union, List
import numpy as np


def stochastic_oscillator(
                            closing_data: Union["np.ndarray", List[float]],
                            high_data: Union["np.ndarray", List[float]],
                            low_data: Union["np.ndarray", List[float]],
                            period: int
                        ) -> "np.ndarray":
    if len(closing_data) < period:
        raise ValueError("Unable to calculate stochastic oscillator indicator for data of length %d"
                         "and period of %d" % (len(closing_data), period))
    ret_indicator = np.zeros(((len(closing_data) - period) + 1, ))
    inf_indicies = []
    for i in range(len(ret_indicator)):
        high = np.max(high_data[i:i+period])
        low = np.min(low_data[i:i+period])
        close = closing_data[i+period-1]
        if high == low:
            ret_indicator[i] = np.inf
            inf_indicies.append(i)
        else:
            ret_indicator[i] = ((close - low) / (high - low)) * 100

    for inf_index in inf_indicies:
        if inf_index == 0:
            ret_indicator[inf_index] = 0
        elif inf_index == len(ret_indicator)-1:
            ret_indicator[inf_index] = ret_indicator[inf_index-1]
        else:
            ret_indicator[inf_index] = (ret_indicator[inf_index-1] + ret_indicator[inf_index+1]) / 2

    return ret_indicator


def ad_oscillator(
                    closing_data: Union["np.ndarray", List[float]],
                    high_data: Union["np.ndarray", List[float]],
                    low_data: Union["np.ndarray", List[float]]
                 ) -> "np.ndarray":
    if len(closing_data) < 2:
        raise ValueError("Unable to calculate accumulation/distribution oscillator indicator for data of length %d"
                         "and period of %d" % (len(closing_data), 2))
    ret_indicator = np.zeros((len(closing_data)-1,), dtype=np.float32)
    for i in range(len(closing_data)-1):
        if high_data[i+1] == low_data[i+1]:
            ret_indicator[i] = 0
        else:
            ret_indicator[i] = (high_data[i+1] - closing_data[i]) / (high_data[i+1] - low_data[i+1])
    return ret_indicator
