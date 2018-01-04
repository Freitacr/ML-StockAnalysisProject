'''
Created on Dec 23, 2017

@author: Colton Freitas, Jim Carey
'''

from .YahooDataFormatting import YahooDataFormatting

class DataFormatter:
    
    def __init__(self, login_credentials, ticker_list):
        self.login_credentials = login_credentials
        self.ticker_list = ticker_list
        
    def getData(self):
        '''
        returns ret_data which is a formatted array with:
        [[Yahoo,[data[]],Google,[data[]]]]
        '''
        ret_data = []
        yah_formatter = YahooDataFormatting(self.ticker_list, self.login_credentials)
        ret_data.extend([yah_formatter.getData()])
        return ret_data