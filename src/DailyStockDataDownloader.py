'''
Created on Dec 19, 2017

@author: Colton Freitas
'''
from StockDataDownloading.DownloaderYahoo import DownloaderYahoo
from StockDataMYSQLUploading.MSYQLDataUploader import MYSQLDataManipulator
from mysql.connector.dbapi import Date
#TODO: Take input from an external source for which stocks to obtain data for
#TODO: Find best way to save password so it isn't readable when not in use
#TODO: Handle duplicate avoidance here after the download phase

if __name__ == '__main__':
    yah = DownloaderYahoo()
    dat, errored = yah.getHistoricalData(['AAPL'])
    dat2, errored = yah.getHistoricalData(['GOOGL'])
    dat.extend(dat2)
    print(len(dat))
    for ticker in dat:
        for day in range(5):
            print(ticker[1][day])
    