from stock_data_analysis_module.data_processing_module.stock_cluster_creator import StockClusterCreator
from stock_data_analysis_module.data_processing_module.data_block_training_data_util import normalizeBlock, splitDataBlock
import datetime as dt
import numpy as np
from typing import List


class StockClusterDataManager:

    def __init__(self, login_credentials: List[str], start_date: str = None,
                 end_date: str = None, num_similar_stocks: int = 12, column_list: List[str] = None):
        if start_date is None:
            start = dt.datetime.now()
            start -= dt.timedelta(weeks=52 * 4)
            start_date = start.isoformat()[:10].replace('-', '/')
        if end_date is None:
            end = dt.datetime.now()
            end_date = end.isoformat()[:10].replace('-', '/')
        self.clusterCreator = StockClusterCreator(
            login_credentials, start_date, end_date, num_similar_stocks, column_list
        )
        availableTickers = [x for x in self.clusterCreator.dataRetriever.data_sources.keys()]
        self.clusters = self.clusterCreator.createClusters(availableTickers)
        self.clusterCreator.close()

    def retrieveTrainingData(self, numDaysPerExample=20, expectation_columns=None):
        ret_x_train, ret_y_train, ret_x_test, ret_y_test = [], [], [], []
        for c in self.clusters:
            data_block = c.retrieveDataBlock()
            normalizeBlock(data_block)
            x_train, y_train, x_test, y_test = splitDataBlock(data_block,
                                                              numDaysPerExample,
                                                              expectation_columns=expectation_columns)
            ret_x_train.extend(x_train)
            ret_y_train.extend(y_train)
            ret_x_test.extend(x_test)
            ret_y_test.extend(y_test)
        return np.array(ret_x_train), np.array(ret_y_train), np.array(ret_x_test), np.array(ret_y_test)

    def retrieveTrainingDataSplit(self, numDaysPerExample=20, expectation_columns=None):
        retMap = {}
        for c in self.clusters:
            dataBlock = c.retrieveDataBlock()
            normalizeBlock(dataBlock)
            x_train, y_train, x_test, y_test = splitDataBlock(dataBlock, numDaysPerExample, expectation_columns=expectation_columns)
            retMap[c.mainTicker] = [x_train, y_train, x_test, y_test]
        return retMap

    def retrieveTrainingDataMovementTargets(self, numDaysPerExample=20, expectation_columns=None):
        x_train, y_train, x_test, y_test = self.retrieveTrainingData(
            numDaysPerExample, expectation_columns=expectation_columns)
        y_train_temp = np.zeros((y_train.shape[0], 3))
        y_test_temp = np.zeros((y_test.shape[0], 3))
        expectation_index = 0
        if expectation_columns is not None:
            expectation_index = expectation_columns[0]
        for i in range(len(x_train)):
            x_test_var = x_train[i][0][-1]
            if y_train[i][0] > x_test_var[expectation_index]:
                y_train_temp[i][2] = 1
            elif y_train[i][0] == x_test_var[expectation_index]:
                y_train_temp[i][1] = 1
            else:
                y_train_temp[i][0] = 1

        for i in range(len(x_test)):
            x_test_var = x_test[i][0][-1]
            if y_test[i][0] > x_test_var[expectation_index]:
                y_test_temp[i][2] = 1
            elif y_test[i][0] == x_test_var[expectation_index]:
                y_test_temp[i][1] = 1
            else:
                y_test_temp[i][0] = 1

        return x_train, y_train_temp, x_test, y_test_temp

    def retrieveTrainingDataMovementTargetsSplit(self, numDatsPerExample=20, expectation_columns=None):
        retMap = self.retrieveTrainingDataSplit(numDatsPerExample, expectation_columns=expectation_columns)
        for ticker, data in retMap.items():
            x_train = data[0]
            y_train = data[1]
            x_test = data[2]
            y_test = data[3]
            y_train_temp = np.zeros((y_train.shape[0], 3))
            y_test_temp = np.zeros((y_test.shape[0], 3))
            expectation_index = 0
            if expectation_columns is not None:
                expectation_index = expectation_columns[0]
            for i in range(len(x_train)):
                x_test_var = x_train[i][0][-1]
                if y_train[i][0] > x_test_var[expectation_index]:
                    y_train_temp[i][2] = 1
                elif y_train[i][0] == x_test_var[expectation_index]:
                    y_train_temp[i][1] = 1
                else:
                    y_train_temp[i][0] = 1

            for i in range(len(x_test)):
                x_test_var = x_test[i][0][-1]
                if y_test[i][0] > x_test_var[expectation_index]:
                    y_test_temp[i][2] = 1
                elif y_test[i][0] == x_test_var[expectation_index]:
                    y_test_temp[i][1] = 1
                else:
                    y_test_temp[i][0] = 1
            retMap[ticker] = (x_train, y_train_temp, x_test, y_test_temp)
        return retMap