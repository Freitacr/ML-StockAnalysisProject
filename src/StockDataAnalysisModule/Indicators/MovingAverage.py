from typing import Union, List
import numpy as np


def EMA(data: Union["np.ndarray", List[float]], period: int) -> "np.ndarray":
    raise NotImplementedError("As EMA is a recursive formula, this implementation will be revisited")
    # if len(data) < period:
    #     raise ValueError("Unable to calculate EMA for data of length %d and a period of %d" %
    #                      (len(data), period))
    # ret_ema = np.zeros(((len(data) - period) + 1))
    #
    # # calculate first EMA entry
    # ret_ema[0] = data[0]  # one initialization technique is to simply use the first data entry
    # # use first EMA to calculate the rest of the values


def SMA(data: Union["np.ndarray", List[float]], period: int) -> "np.ndarray":
    if len(data) < period:
        raise ValueError("Unable to calculate SMA for data of length %d and a period of %d" %
                         (len(data), period))
    ret_sma = np.zeros(((len(data) - period) + 1, ))
    for i in range(len(ret_sma)):
        ret_sma[i] = np.average(data[i:i+period])
    return ret_sma
