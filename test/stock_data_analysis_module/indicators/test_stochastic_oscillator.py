import unittest

import numpy as np

from src.stock_data_analysis_module.indicators import stochastic_oscillator


class StochasticOscillatorTestCase(unittest.TestCase):

    def __init__(self, *args):
        super().__init__(*args)
        self._empty_sequence = []
        self._empty_ndarray = np.zeros((0,))

    def test_invalid_length(self):
        with self.assertRaises(ValueError):
            stochastic_oscillator.stochastic_oscillator(self._empty_sequence,
                                                        self._empty_sequence,
                                                        self._empty_sequence,
                                                        period=10)
        with self.assertRaises(ValueError):
            stochastic_oscillator.stochastic_oscillator(self._empty_ndarray,
                                                        self._empty_ndarray,
                                                        self._empty_ndarray,
                                                        period=10)


if __name__ == '__main__':
    unittest.main()
