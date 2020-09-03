from typing import Union, List

import numpy as np
import pandas


def EMA(data: Union["np.ndarray", List[float]], period: int) -> "np.ndarray":
    df = pandas.DataFrame(data)
    ema = df.ewm(span=period, adjust=False)
    return ema.obj.to_numpy()[period:]


def SMA(data: Union["np.ndarray", List[float]], period: int) -> "np.ndarray":
    if len(data) < period:
        raise ValueError("Unable to calculate SMA for data of length %d and a period of %d" %
                         (len(data), period))
    ret_sma = np.zeros(((len(data) - period) + 1, ))
    for i in range(len(ret_sma)):
        ret_sma[i] = np.average(data[i:i+period])
    return ret_sma


def WMA(data: Union["np.ndarray", List[float]], period: int) -> "np.ndarray":
    if len(data) < period:
        raise ValueError("Unable to calculate WMA for data of length %d and a period of %d" %
                         (len(data), period))
    ret_wma = np.zeros(((len(data) - period) + 1,))
    for i in range(len(ret_wma)):
        averaging_data = data[i:i+period]
        weighted_sum = 0
        weighted_scale = 0
        for j in range(len(averaging_data)):
            weighted_sum += averaging_data[j] * (j+1)
            weighted_scale += j+1
        ret_wma[i] = weighted_sum / weighted_scale
    return ret_wma
