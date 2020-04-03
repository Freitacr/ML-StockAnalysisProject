import numpy as np

def normalizeBlock(trainingBlock):
    for dataRow in trainingBlock:
        dataMax = max(dataRow)
        dataMin = min(dataRow)
        for i in range(len(dataRow)):
            dataRow[i] -= dataMin
        dataRow /= dataMax

def splitDataBlock(trainingBlock, numDaysPerExample, trainingPercentage = .8):
    maxTrainingIndex = int(len(trainingBlock[0]) * trainingPercentage) - numDaysPerExample
    x_train, y_train = [],[]
    x_test, y_test = [],[]
    i = 0
    for i in range(maxTrainingIndex):
        x_data = []
        for dataRow in trainingBlock:
            x_data.append(dataRow[i:i+numDaysPerExample])
        x_data = np.array(x_data)
        y_data = np.array([trainingBlock[0][i+numDaysPerExample]])
        x_train.append(x_data)
        y_train.append(y_data)
    
    for i in range(maxTrainingIndex, len(trainingBlock[0])-numDaysPerExample):
        x_data = []
        for dataRow in trainingBlock:
            x_data.append(dataRow[i:i+numDaysPerExample])
        x_data = np.array(x_data)
        y_data = np.array([trainingBlock[0][i+numDaysPerExample]])
        x_test.append(x_data)
        y_test.append(y_data)
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)