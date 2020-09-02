import unittest

import numpy as np

from src.stock_data_analysis_module.indicators import momentum


class MomentumTestCase(unittest.TestCase):

    def __init__(self, *args):
        super().__init__(*args)
        self._empty_sequence = []
        self._empty_ndarray = np.zeros((0,))
        self._example_a = [0, 1, 2, 3, 4, 5]
        self._example_b = [1, -1, 1, -1, 1, -1]
        self._a_2_result = [2, 2, 2, 2]
        self._b_2_result = [0, 0, 0, 0]
        self._a_3_result = [3, 3, 3]

    def test_invalid_length(self):
        with self.assertRaises(ValueError):
            momentum.momentum(self._empty_ndarray, 10)
        with self.assertRaises(ValueError):
            momentum.momentum(self._empty_sequence, 10)

    def test_standard_example(self):
        result = momentum.momentum(self._example_a, period=2)
        if len(result) != len(self._a_2_result):
            self.fail("Unexpected results from momentum calculation with "
                      "the sequence %s.\nExpected: %s\nActual: %s" %
                      (str(self._example_a), str(self._a_2_result), str(result)))
        for i in range(len(result)):
            if result[i] != self._a_2_result[i]:
                self.fail("Unexpected results from momentum calculation with "
                          "the sequence %s.\nExpected: %s\nActual: %s" %
                          (str(self._example_a), str(self._a_2_result), str(result)))

    def test_standard_example_2(self):
        result = momentum.momentum(self._example_b, period=2)
        if len(result) != len(self._b_2_result):
            self.fail("Unexpected results from momentum calculation with "
                      "the sequence %s.\nExpected: %s\nActual: %s" %
                      (str(self._example_b), str(self._b_2_result), str(result)))
        for i in range(len(result)):
            if result[i] != self._b_2_result[i]:
                self.fail("Unexpected results from momentum calculation with "
                          "the sequence %s.\nExpected: %s\nActual: %s" %
                          (str(self._example_b), str(self._b_2_result), str(result)))

    def test_standard_example_3(self):
        result = momentum.momentum(self._example_a, period=3)
        if len(result) != len(self._a_3_result):
            self.fail("Unexpected results from momentum calculation with "
                      "the sequence %s.\nExpected: %s\nActual: %s" %
                      (str(self._example_a), str(self._a_3_result), str(result)))
        for i in range(len(result)):
            if result[i] != self._a_3_result[i]:
                self.fail("Unexpected results from momentum calculation with "
                          "the sequence %s.\nExpected: %s\nActual: %s" %
                          (str(self._example_a), str(self._a_3_result), str(result)))


if __name__ == '__main__':
    unittest.main()
