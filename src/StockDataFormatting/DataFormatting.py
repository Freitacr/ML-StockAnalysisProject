'''
Created on Dec 23, 2017

@author: Colton Freitas
'''

from .YahooDataFormatting import YahooDataFormatting

class DataFormatter:
    
    def __init__(self, login_credentials, ticker_list):
        self.login_credentials = login_credentials
        self.ticker_list = ticker_list
        
    def getData(self):
        ret_data = []
        yah_formatter = YahooDataFormatting(self.ticker_list, self.login_credentials)
        ret_data.extend([yah_formatter.getData()])
        return ret_data