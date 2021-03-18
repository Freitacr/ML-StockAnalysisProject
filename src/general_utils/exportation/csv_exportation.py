from typing import List, Tuple


def export_predictions(ticker_data: List[Tuple[str, str, float]], out_path: str):
    with open(out_path, 'w') as open_file:
        open_file.write(','.join(['', '', '', '', '']) + '\n')
        open_file.write(','.join(['', 'Date', 'Ticker', 'Prediction', 'Observed Accuracy']) + '\n')
        column_format = ",,{0},{1},{2:.2f}%\n"
        for ticker, prediction, accuracy in ticker_data:
            open_file.write(column_format.format(ticker, prediction, accuracy*100))
