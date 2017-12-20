'''
Created on Dec 19, 2017

@author: Colton Freitas
'''
from StockDataDownloading.DownloaderYahoo import DownloaderYahoo
from StockDataMYSQLUploading.MYSQLUtils.MYSQLUtils import connect as MYSQLConnect
#TODO: Use uploading class to upload to the MYSQL database
#TODO: Take input from an external source for which stocks to obtain data for


def testMYSQL():
    sqlConnection = MYSQLConnect("localhost", "user", "Sora1674@", "conn_test")
    return sqlConnection


if __name__ == '__main__':
    yahooDownloader = DownloaderYahoo()
    ticker_list = ["AAPL", "GOOGL", "XYZZYX"] #Temporary testing list. The last value is intentionally wrong
    data, errored = yahooDownloader.getHistoricalData(ticker_list)
    
    testMYSQL()
    