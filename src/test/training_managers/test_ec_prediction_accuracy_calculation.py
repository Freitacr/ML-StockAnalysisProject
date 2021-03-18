import unittest

import numpy as np

from training_managers.evolutionary_computation_trainer import prediction_truth_calculation, \
    extract_accuracy_from_prediction_truths


class EvolutionaryComputationAccuracyCalculationTestCase(unittest.TestCase):

    def test_ec_prediction_accuracy_calculation(self):
        num_days_per_predictions = 2

        predictions = [
            [[True, True], [True, True], [True, True]],
            [[False, True], [False, True], [False, True]],
            [[True, True], [True, True], [True, True]],
            [[False, False], [False, False], [False, False]],
            [[False, False], [False, False], [False, False]],
            [[True, True], [True, True], [True, True]]
        ]

        closing_prices = [
            10,
            12,
            8,
            20,
            5,
            3
        ]

        expected_truths = np.array([
            [[True, True], [True, True], [True, True]],
            [[True, True], [True, True], [True, True]],
            [[True, True], [True, True], [True, True]],
            [[True, False], [True, False], [True, False]]
        ])

        # expected_accuracies = [.75, 1]

        prediction_truths = prediction_truth_calculation(predictions, closing_prices,
                                                         num_days_per_prediction=num_days_per_predictions)

        for index, truths in enumerate(prediction_truths):
            for prediction_index, prediction_truth in enumerate(truths):
                for i in range(len(prediction_truth)):
                    expected_truth = expected_truths[index][prediction_index][i]
                    truth = prediction_truth[i]
                    self.assertEqual(expected_truth, truth,
                                     "\n\nPrediction truths vs expected truths:\n"
                                     f"{prediction_truth}\n{expected_truths[index][prediction_index]}\n"
                                     "These truths were based on these prices: "
                                     f"{closing_prices[index: index + num_days_per_predictions + 1]}"
                                     )

    def test_ec_accuracy_extraction(self):
        truths = np.array([
            [[True, True], [True, True], [True, True]],
            [[True, True], [True, True], [True, True]],
            [[True, True], [True, True], [True, True]],
            [[True, False], [True, False], [True, False]]
        ])
        expected_accuracies = [[1.0, .75], [1.0, .75], [1.0, .75]]

        extracted_accuracies = extract_accuracy_from_prediction_truths(truths)
        for index, accuracies in enumerate(extracted_accuracies):
            for i in range(len(accuracies)):
                self.assertEqual(expected_accuracies[index][i], accuracies[i],
                                 "\n\nExpected accuracies vs predicted accuracies:\n"
                                 f"{expected_accuracies[index]}\n{accuracies}\n"
                                 "These accuracies were based off of these prediction truths:\n"
                                 f"{truths[:, index]}")


if __name__ == '__main__':
    unittest.main()
