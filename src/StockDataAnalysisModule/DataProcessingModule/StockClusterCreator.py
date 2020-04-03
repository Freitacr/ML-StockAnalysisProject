from GeneralUtils.DataStorageClasses.StockCluster import StockCluster
from StockDataAnalysisModule.DataProcessingModule.DataRetrievalModule.RangedDataRetriever import RangedDataRetriever
import numpy as np

def ensureDataLengthSanityEnsurance(tickerData):
    lastHistDate = tickerData[0][-1][0]
    maxLength = len(tickerData[0])
    for dataList in tickerData:
        if len(dataList) == 0:
            continue
        if not dataList[-1][0] == lastHistDate:
            raise ValueError("Data did not end on the same date, recovery impossible")
        if len(dataList) > maxLength:
            maxLength = len(dataList)
    
    #todo:
    #go element by element ensuring that all dates are matched
    #if not, then the date's data becomes the average of the date preceding and following it.
    
    toRemove = []

    for i in range(len(tickerData)):
        dataList = tickerData[i]
        if not len(dataList) == maxLength:
            toRemove.append(i)

    return toRemove

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
        for j in range(n):
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

    def __init__(self, login_credentials, startDate, endDate, similar_tickers = 5):
        self.dataRetriever = RangedDataRetriever(login_credentials, ['hist_date', 'adj_close'], startDate, endDate)
        self.similarTickers = similar_tickers
        
    def createClusters(self, tickerList):
        #retrieve data for tickers
        tickerData = [self.dataRetriever.retrieveData(ticker) for ticker in tickerList]
        
        #ensure data all has same length
        toRemoveIndices = ensureDataLengthSanityEnsurance(tickerData)
        toRemoveTickers = [tickerList[i] for i in toRemoveIndices]
        toRemoveData = [tickerData[i] for i in toRemoveIndices]
        for toRem in toRemoveTickers:
            tickerList.remove(toRem)
        for toRem in toRemoveData:
            tickerData.remove(toRem)

        #have numpy calculate correlation coefficients
        tickerData = [ [y[1] for y in x] for x in tickerData]
        coefficients = np.corrcoef(tickerData)
        
        #grab top n correficients for each stock
        coeffPairs = retrieveTopNStrongestCoefficients(coefficients, self.similarTickers)
        
        #construct clusters from top n coefficients for each ticker
        retClusters = []
        for i in range(len(coeffPairs)):
            retClusters.append(constructCluster(coeffPairs[i], tickerList, tickerData))
        
        return retClusters

    def close(self):
        self.dataRetriever.close()