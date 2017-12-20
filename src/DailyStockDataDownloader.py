'''
Created on Dec 19, 2017

@author: Colton Freitas
'''
from StockDataDownloading.DownloaderYahoo import DownloaderYahoo

#TODO: Use uploading class to upload to the MYSQL database
#TODO: Take input from an external source for which stocks to obtain data for

if __name__ == '__main__':
    yahooDownloader = DownloaderYahoo()
    ticker_list = ["AAPL", "GOOGL", "XYZZYX"] #Temporary testing list. The last value is intentionally wrong
    data, errored = yahooDownloader.getHistoricalData(ticker_list)