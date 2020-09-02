from typing import Union, List
import numpy as np


def momentum(data: Union["np.ndarray", List[float]], period: int) -> "np.ndarray":
    if len(data) < period:
        raise ValueError("Unable to calculate momentum for data of length %d and a period of %d" %
                         (len(data), period))
    ret_momentum = np.zeros((len(data) - period))
    for i in range(len(ret_momentum)):
        ret_momentum[i] = data[i+period] - data[i]
    return ret_momentum
