import numpy as np

class StockCluster:

    def __init__(self, mainTicker, mainData, supportingTickers, supportingData):
        self.tickerAssociations = {}
        self.tickerAssociations[mainTicker] = mainData
        self.mainTicker = mainTicker
        for i in range(len(supportingTickers)):
            ticker = supportingTickers[i]
            data = supportingData[i]
            self.tickerAssociations[ticker] = data

    def retrieveDataBlock(self):
        retData = []
        keys = sorted(self.tickerAssociations.keys())
        keys.remove(self.mainTicker)
        retData.append(self.tickerAssociations[self.mainTicker])
        for key in keys:
            retData.append(self.tickerAssociations[key])
        return np.array(retData)