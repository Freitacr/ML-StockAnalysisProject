from general_utils.data_storage_classes.stock_cluster import StockCluster
from stock_data_analysis_module.data_processing_module.data_retrieval_module.ranged_data_retriever import RangedDataRetriever
import numpy as np
from datetime import date, datetime


def date_to_timestamp(date_in: date):
    return datetime.fromisoformat(date_in.isoformat()).timestamp()


def ensure_data_length_consistency(ticker_data):
    last_hist_date = ticker_data[0][-1][0]
    max_length = len(ticker_data[0])
    for data_list in ticker_data:
        if len(data_list) == 0:
            continue
        if not data_list[-1][0] == last_hist_date:
            raise ValueError("Data did not end on the same date, recovery impossible")
        if len(data_list) > max_length:
            max_length = len(data_list)

    # todo:
    # go element by element ensuring that all dates are matched
    # if not, then the date's data becomes the average of the date preceding and following it.
    
    to_remove = []

    for i in range(len(ticker_data)):
        data_list = ticker_data[i]
        if not len(data_list) == max_length:
            to_remove.append(i)

    return to_remove


def getStrongestCoefficient(coeffList, currIndex):
    maxCoeff = -2
    maxIndex = -1
    for i in range(len(coeffList)):
        if i == currIndex:
            continue
        if abs(coeffList[i]) > maxCoeff:
            maxCoeff = abs(coeffList[i])
            maxIndex = i
    
    coeffList[maxIndex] = 0
    return maxIndex, maxCoeff


def retrieveTopNStrongestCoefficients(coefficients, n):
    retCoeffPairs = []
    for i in range(len(coefficients)):
        considered = coefficients[i]
        coeffs = []
        for _ in range(n):
            coeffs.append(getStrongestCoefficient(considered, i)[0])
        retCoeffPairs.append((i, coeffs))
    return retCoeffPairs


def constructCluster(coefficientPair, tickerList, tickerData):
    mainTicker = tickerList[coefficientPair[0]]
    mainTickerData = tickerData[coefficientPair[0]]
    supportingTickers = []
    supportingTickerData = []
    for i in range(len(coefficientPair[1])):
        supportTicker = tickerList[coefficientPair[1][i]]
        supportData = tickerData[coefficientPair[1][i]]
        supportingTickers.append(supportTicker)
        supportingTickerData.append(supportData)
    return StockCluster(mainTicker, mainTickerData, supportingTickers, supportingTickerData)


class StockClusterCreator:

    def __init__(self, login_credentials, start_date, end_date, similar_tickers=5, columns=None):
        column_list = None
        if columns is None:
            column_list = ['hist_date', 'adj_close']
        else:
            # this is intentional to allow duplicates. When a data requestor wants the historical date in
            # their dataset, this duplicate is required as this class removes the first entry from the date
            # it returns
            column_list = ['hist_date']
            for column in columns:
                column_list.append(column)
        self.dataRetriever = RangedDataRetriever(login_credentials, column_list, start_date, end_date)
        self.similarTickers = similar_tickers
        
    def createClusters(self, ticker_list):
        # retrieve data for tickers
        ticker_data = [self.dataRetriever.retrieveData(ticker) for ticker in ticker_list]
        
        # ensure data all has same length
        to_remove_indices = ensure_data_length_consistency(ticker_data)
        to_remove_tickers = [ticker_list[i] for i in to_remove_indices]
        to_remove_data = [ticker_data[i] for i in to_remove_indices]
        for toRem in to_remove_tickers:
            ticker_list.remove(toRem)
        for toRem in to_remove_data:
            ticker_data.remove(toRem)

        # remove the first element from each data entry, as it is the historical date.
        for i in range(len(ticker_data)):
            curr_data = ticker_data[i]
            ticker_data[i] = [x[1:] for x in curr_data]

        # determine whether to shift correlation check by one element due to included dates
        start_index = 0
        if type(ticker_data[0][0][0]) == date:
            start_index = 1

        # have numpy calculate average correlation coefficient
        coefficients = np.zeros((len(ticker_data), len(ticker_data)))
        for i in range(start_index, len(ticker_data[0][0])):
            temp_data = [[y[i] for y in x] for x in ticker_data]
            coefficients += abs(np.corrcoef(temp_data))
        coefficients /= len(ticker_data[0][0])-1
        
        # grab top n coefficients for each stock
        coeff_pairs = retrieveTopNStrongestCoefficients(coefficients, self.similarTickers)
        
        # construct clusters from top n coefficients for each ticker
        ret_clusters = []
        for i in range(len(coeff_pairs)):
            ret_clusters.append(constructCluster(coeff_pairs[i], ticker_list, ticker_data))
        
        return ret_clusters

    def close(self):
        self.dataRetriever.close()
