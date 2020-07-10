'''
Created on Dec 23, 2017

@author: Colton Freitas, Jim Carey
'''


import abc
from typing import List, Set


class DataFormatterRegistry:
    
    def __init__(self):
        '''Initialization method
        @param ticker_list: List of stock tickers to obtain and format data for
        @param login_credentials: Login credentials for the MySQL Server 
        '''
        self.formatter_registry: Set[DataFormatter] = set()
        
    def getData(self, ticker_list):
        '''Obtain and format data on all tickers from self.ticker_list
        @return List formatted as such [ [datasource1, [ [ticker, [day1data, day2data...] ], [ticker2 ....] ] ], [datasourceN ... ] ]
        '''
        ret_data = []
        for formatter in self.formatter_registry:
            ret_data.append(formatter.get_data(ticker_list))
        return ret_data


class DataFormatter(abc.ABC):

    @abc.abstractmethod
    def get_data(self, ticker_list: List[str]):
        pass


registry = DataFormatterRegistry()
