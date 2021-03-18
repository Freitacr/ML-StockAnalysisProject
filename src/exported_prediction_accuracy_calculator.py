from typing import TextIO
from datetime import datetime
import os
from os import path
import sys

import numpy as np

from general_utils.logging import logger
from general_utils.mysql_management.mysql_tables import stock_data_table
from stock_data_analysis_module.data_processing_module.data_retrieval_module import period_data_retriever


# TODO[Colton Freitas] Add dates into the exported predictions using the dates unknown dates
#   from the prediction phase


def calculate_file_accuracy(file_handle: TextIO,
                            data_retriever: period_data_retriever.PeriodDataRetriever,
                            num_backtrack_days: int = 5):
    predictions = {}
    file_handle.readline()
    file_handle.readline()
    for line in file_handle:
        _, _, ticker, prediction, _ = line.split(',')
        predictions[ticker] = prediction
    date_bound_accuracies = {}
    for i in range(num_backtrack_days):
        correct_predictions = 0
        bound_date = None
        for ticker, prediction in predictions.items():
            stock_data = data_retriever.retrieve_data(ticker, max_rows=num_backtrack_days+1)
            stock_data = np.array(stock_data)
            close = list(reversed(stock_data[:, 0]))
            hist_date = list(reversed(stock_data[:, 1]))
            close = ["Trend Upward" if close[j] >= close[j-1] else "Trend Downward"
                     for j in range(1, len(close))]
            hist_date = hist_date[:-1]
            bound_date = hist_date[-(i+1)]
            close = close[-(i+1)]
            if close == prediction:
                correct_predictions += 1
        date_bound_accuracies[bound_date] = (correct_predictions / len(predictions)) * 100
    return date_bound_accuracies


if __name__ == '__main__':
    args = sys.argv[1:]
    input_directory = args[0]
    out_directory = args[1]
    if not path.exists(input_directory):
        logger.logger.log(logger.FATAL_ERROR, "Input directory does not exist.")
        exit(1)
    if not path.exists(out_directory):
        os.mkdir(input_directory)
    prediction_files = []
    for file in os.listdir(input_directory):
        if file.endswith('.csv') and not file.startswith('amalg'):
            prediction_files.append(input_directory + path.sep + file)

    data_retriever = period_data_retriever.PeriodDataRetriever(
        [
            stock_data_table.CLOSING_PRICE_COLUMN_NAME,
            stock_data_table.HISTORICAL_DATE_COLUMN_NAME
        ],
        datetime.now().isoformat()[:10].replace('-', '/')
    )
    out_string = ""
    out_file_header_template = "Dated Accuracy for file {0}:\n"
    out_file_accuracy_template = "{0}: {1:.2f}\n"
    for file in prediction_files:
        with open(file, 'r') as open_file:
            date_accuracies = calculate_file_accuracy(open_file, data_retriever, num_backtrack_days=30)
            out_string += out_file_header_template.format(file)
            for date, accuracy in date_accuracies.items():
                out_string += out_file_accuracy_template.format(date, accuracy)
    logger.logger.log(logger.OUTPUT, "\n" + out_string)
