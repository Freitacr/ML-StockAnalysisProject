'''
Created on Dec 23, 2017

@author: Colton Freitas, Jim Carey
'''

from .YahooDataFormatting import YahooDataFormatting

class DataFormatter:
    
    def __init__(self, login_credentials, ticker_list):
        '''Initialization method
        @param ticker_list: List of stock tickers to obtain and format data for
        @param login_credentials: Login credentials for the MySQL Server 
        '''
        self.login_credentials = login_credentials
        self.ticker_list = ticker_list
        
    def getData(self):
        '''Obtain and format data on all tickers from self.ticker_list
        @return List formatted as such [ [datasource1, [ [ticker, [day1data, day2data...] ], [ticker2 ....] ] ], [datasourceN ... ] ]
        '''
        ret_data = []
        yah_formatter = YahooDataFormatting(self.ticker_list, self.login_credentials)
        ret_data.extend([yah_formatter.getData()])
        return ret_data