from typing import Union, List
import numpy as np


def rate_of_change(close_data: Union["np.ndarray", List[float]], period: int) -> np.ndarray:
    if len(close_data) < period:
        raise ValueError("Unable to calculate rate of change for data of length %d and a period of %d" %
                         (len(close_data), period))
    ret_rate_of_change = np.zeros((len(close_data) - period,))
    for i in range(len(ret_rate_of_change)):
        ret_rate_of_change[i] = (close_data[i + period] - close_data[i]) / close_data[i]
    ret_rate_of_change *= 100
    return ret_rate_of_change
