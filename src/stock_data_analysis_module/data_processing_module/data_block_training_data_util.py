import numpy as np
from datetime import datetime, date


def date_to_timestamp(date_in: date):
    return datetime.fromisoformat(date_in.isoformat()).timestamp()


def normalizeBlock(training_block):
    for dataRow in training_block:
        for i in range(len(dataRow[0])):
            if type(dataRow[0][i]) == date:
                max_stamp = datetime.now().timestamp()
                for j in range(len(dataRow)):
                    dataRow[j][i] = date_to_timestamp(dataRow[j][i]) / max_stamp
            else:
                elements = [x[i] for x in dataRow]
                data_max = max(elements)
                data_min = min(elements)
                for j in range(len(dataRow)):
                    dataRow[j][i] = (dataRow[j][i] - data_min) / data_max


def splitDataBlock(trainingBlock, numDaysPerExample, trainingPercentage = .8, expectation_columns=None):
    max_training_index = int(len(trainingBlock[0]) * trainingPercentage) - numDaysPerExample
    x_train, y_train = [], []
    x_test, y_test = [], []
    for i in range(max_training_index):
        x_data = []
        for dataRow in trainingBlock:
            x_data.append(dataRow[i:i+numDaysPerExample])
        x_data = np.array(x_data)
        y_data = np.array([trainingBlock[0][i+numDaysPerExample]])
        if expectation_columns is not None:
            y_data = [[y[x] for x in expectation_columns] for y in y_data]
        x_train.append(x_data)
        y_train.append(y_data)
    
    for i in range(max_training_index, len(trainingBlock[0])-numDaysPerExample):
        x_data = []
        for dataRow in trainingBlock:
            x_data.append(dataRow[i:i+numDaysPerExample])
        x_data = np.array(x_data)
        y_data = np.array([trainingBlock[0][i+numDaysPerExample]])
        if expectation_columns is not None:
            y_data = [[y[x] for x in expectation_columns] for y in y_data]
        x_test.append(x_data)
        y_test.append(y_data)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)