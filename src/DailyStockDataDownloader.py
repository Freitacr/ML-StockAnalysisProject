'''
Created on Dec 19, 2017

@author: Colton Freitas
'''
from StockDataDownloading.DownloaderYahoo import DownloaderYahoo
from StockDataMYSQLManagement.MSYQLDataManipulator import MYSQLDataManipulator
from mysql.connector.dbapi import Date
from datetime import datetime as dt
from StockDataFormatting.YahooDataFormatting import YahooDataFormatting
#TODO: Take input from an external source for which stocks to obtain data for
#TODO: Save and load login credentials from an external file. As this will not be visible to end users, there is no worry about the security of the file
#TODO: Handle duplicate avoidance here after the download phase

if __name__ == '__main__':
    login_credentials = ['localhost', 'root', 'Sora1674@', 'stock_testing']
    yahFormatter = YahooDataFormatting(['AAPL', 'GOOGL'], login_credentials)
    data = yahFormatter.getData()
    
    #data returned in format ['yahoo', [['ticker', [data]]]]
    print(len(data))
    for day in data[1][0][1]:
        print(day.rstrip())
    #for ticker in data[1]:
    #    print(ticker[0])
    
    #yah = DownloaderYahoo()
    #yesterday = dt.strptime("2017-12-21", "%Y-%m-%d")
    #dat, errored = yah.getHistoricalData(['AAPL'], max_number_of_days= 0, start_date = yesterday)
    #print(len(dat))
    #for ticker in dat:
    #    for day in range(5):
    #        print(ticker[1][day])
    