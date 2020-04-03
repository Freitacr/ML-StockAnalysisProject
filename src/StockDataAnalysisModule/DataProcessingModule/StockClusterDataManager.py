from StockDataAnalysisModule.DataProcessingModule.StockClusterCreator import StockClusterCreator
from StockDataAnalysisModule.DataProcessingModule.DataBlockTrainingDataUtil import normalizeBlock, splitDataBlock
import datetime as dt
import numpy as np

class StockClusterDataManager:

    def __init__(self, login_credentials, startDate = None, endDate = None, numSimilarStocks = 12):
        if startDate == None:
            start = dt.datetime.now()
            start -= dt.timedelta(weeks=52 * 4)
            startDate = start.isoformat()[:10].replace('-', '/')
        if endDate == None:
            end = dt.datetime.now()
            endDate = end.isoformat()[:10].replace('-', '/')
        self.clusterCreator = StockClusterCreator(login_credentials, startDate, endDate, numSimilarStocks)
        availableTickers = [x for x in self.clusterCreator.dataRetriever.data_sources.keys()]
        self.clusters = self.clusterCreator.createClusters(availableTickers)
        self.clusterCreator.close()

    def retrieveTrainingData(self, numDaysPerExample=20, normalized=True):
        ret_x_train, ret_y_train, ret_x_test, ret_y_test = [],[],[],[]
        for c in self.clusters:
            dataBlock = c.retrieveDataBlock()
            if normalized:
                normalizeBlock(dataBlock)
            x_train, y_train, x_test, y_test = splitDataBlock(dataBlock, numDaysPerExample)
            ret_x_train.extend(x_train)
            ret_y_train.extend(y_train)
            ret_x_test.extend(x_test)
            ret_y_test.extend(y_test)
        return np.array(ret_x_train), np.array(ret_y_train), np.array(ret_x_test), np.array(ret_y_test)

    def retrieveTrainingDataSplit(self, numDaysPerExample=20, normalized=True):
        retMap = {}
        for c in self.clusters:
            dataBlock = c.retrieveDataBlock()
            if normalized:
                normalizeBlock(dataBlock)
            x_train, y_train, x_test, y_test = splitDataBlock(dataBlock, numDaysPerExample)
            retMap[c.mainTicker] = [x_train, y_train, x_test, y_test]
        return retMap

    def retrieveTrainingDataMovementTargets(self, numDaysPerExample=20, normalized=True):
        x_train, y_train, x_test, y_test = self.retrieveTrainingData(numDaysPerExample, normalized=normalized)
        for i in range(len(x_train)):
            x_test_var = x_train[i][0][-1]
            if y_train[i] > x_test_var:
                y_train[i] = 2
            elif y_train[i] == x_test_var:
                y_train[i] = 1
            else:
                y_train[i] = 0

        for i in range(len(x_test)):
            x_test_var = x_test[i][0][-1]
            if y_test[i] > x_test_var:
                y_test[i] = 2
            elif y_test[i] == x_test_var:
                y_test[i] = 1
            else:
                y_test[i] = 0

        return x_train, y_train, x_test, y_test

    def retrieveTrainingDataMovementTargetsSplit(self, numDatsPerExample=20, normalized=True):
        retMap = self.retrieveTrainingDataSplit(numDatsPerExample, normalized=normalized)
        for ticker, data in retMap.items():
            x_train = data[0]
            y_train = data[1]
            x_test = data[2]
            y_test = data[3]
            for i in range(len(x_train)):
                x_test_var = x_train[i][0][-1]
                if y_train[i] > x_test_var:
                    y_train[i] = 2
                elif y_train[i] == x_test_var:
                    y_train[i] = 1
                else:
                    y_train[i] = 0

            for i in range(len(x_test)):
                x_test_var = x_test[i][0][-1]
                if y_test[i] > x_test_var:
                    y_test[i] = 2
                elif y_test[i] == x_test_var:
                    y_test[i] = 1
                else:
                    y_test[i] = 0
        return retMap