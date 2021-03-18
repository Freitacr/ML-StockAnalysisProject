import os
from os import path


def amalgamate_csvs(csv_dir: str):
    amalgamation_candidates = []
    for file in os.listdir(csv_dir):
        if file.endswith('csv') and file != 'amalgamation.csv':
            amalgamation_candidates.append(csv_dir + path.sep + file)
    ticker_predictions = {}
    for file in amalgamation_candidates:
        with open(file, 'r') as open_file:
            open_file.readline()
            open_file.readline()
            for line in open_file:
                _, _, ticker, prediction, _ = line.split(',')
                if ticker not in ticker_predictions:
                    ticker_predictions[ticker] = {}
                if prediction not in ticker_predictions[ticker]:
                    ticker_predictions[ticker][prediction] = 0
                ticker_predictions[ticker][prediction] += 1
    with open(csv_dir + path.sep + 'amalgamation.csv', 'w') as open_file:
        for ticker, prediction_dict in ticker_predictions.items():
            total_val = 0
            for _, val in prediction_dict.items():
                total_val += val
            open_file.write(f'{ticker},,{total_val}\n')
            for prediction, val in prediction_dict.items():
                open_file.write(',{0},{1:.2f}\n'.format(prediction, val/total_val))
            open_file.write('\n')
