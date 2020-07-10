'''
Created on Dec 23, 2017

@author: Colton Freitas, Jim Carey
'''


import abc
from typing import List, Set, Iterable, Any, Tuple


class TickerFormattedData(object):

    def __init__(self, downloaded_ticker: str, downloaded_data: List[Iterable[Any]]):
        self.ticker = downloaded_ticker
        self.data = downloaded_data


class DataFormatterRegistry:

    def __init__(self):
        '''Initialization method
        @param ticker_list: List of stock tickers to obtain and format data for
        '''
        self.formatter_registry: Set[DataFormatter] = set()

    def get_data(self, ticker_list) -> List[Tuple[str, Iterable[TickerFormattedData]]]:
        '''Obtain and format data on all tickers from self.ticker_list
        '''
        ret_data = []
        for formatter in self.formatter_registry:
            ret_data.append(formatter.get_data(ticker_list))
        return ret_data


class DataFormatter(abc.ABC):

    @abc.abstractmethod
    def get_data(self, ticker_list: List[str]) -> Tuple[str, Iterable[TickerFormattedData]]:
        pass


registry = DataFormatterRegistry()
