'''
Created on Dec 24, 2017

@author: Colton Freitas
'''

from GeneralUtils.StockDataMYSQLManagement.MSYQLDataManipulator import MYSQLDataManipulator
from asyncio.futures import InvalidStateError

#TODO: Allow option to use a conditional in the select_from_table method in __retrieveData
#As this class is subject to more changes, the documentation will be flushed out more when its more complete

class DataRetriever:
    
    def __init__ (self, login_credentials, column_list):
        self.data_man = MYSQLDataManipulator(login_credentials[0], login_credentials[1], login_credentials[2], login_credentials[3])
        self.column_list = column_list
        self.__setupStockSourceList()
        self.__retrieveData()
        self.data_man.close(commit = False)
    
    def __setupStockSourceList(self):
        #Right now this is going to be a lot more temporary, just using whatever source has actual data on the stock
        #The sources are also going to be in order of preference of use (Yahoo, then Google right now. Even though Google has no data
        #Downloading yet)
        source_availability = self.data_man.select_from_table('stock_list', ['ticker', 'yahoo', 'google'])
        tickers = [x[0] for x in source_availability]
        yahoo_availability = [x[1] for x in source_availability]
        google_availability = [x[2] for x in source_availability]
        self.data_sources = []
        for index in range(len(tickers)):
            if yahoo_availability[index]:
                self.data_sources.extend([(tickers[index], 'yahoo')])
            elif google_availability[index]:
                self.data_sources.extend([(tickers[index], 'google')])
            else:
                raise InvalidStateError("%s has no data, but exists in the database. This is bad." % tickers[index])
    
    def __retrieveData(self):
        self.retrieved_data = []
        for ticker, source in self.data_sources:
            self.retrieved_data.extend([[ticker, self.data_man.select_from_table("%s_%s_data" % (ticker, source), self.column_list)]])
        
    def getData(self):
        return self.retrieved_data