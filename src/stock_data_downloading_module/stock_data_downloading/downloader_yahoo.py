'''
Created on Dec 19, 2017

@author: Colton Freitas, Jim Carey

Intended Purpose: Given a list of stock tickers to download, download the historical
    data for each of the stock tickers. If the ticker is not valid, two more attempts (on a timeout)
    should be made to download data for it. In the event that the ticker still has an error,
    it should be returned within the errored_tickers array.
Intended Purpose: Given a list of stock tickers to download, download the historic data
    for each of the stock tickers. The data and any errored tickers should be returned as separate lists
    The data should originate from Yahoo!, and should deal with Yahoo!'s download links.
'''

from .yahoo_downloading_utils import cookie_manager as cm
from .http_utils.http_utils import openURL
from general_utils.logging import logger
from datetime import datetime, timedelta


class DownloaderYahoo:
    def __init__(self):
        '''Initialization function'''
        self.__urlBase = 'https://query1.finance.yahoo.com/v7/finance/download/{0}?period1={1}&period2={2}&interval=1d&events=history&crumb={3}'
        try:
            self.cookie_man = cm.CookieManager()
        except ValueError as e:
            logger.logger.log(logger.FATAL_ERROR, "Error occurred while obtaining cookie: {0}".format(str(e)))
            logger.logger.log(logger.FATAL_ERROR, "Exiting as continuation is impossible")
            exit(1)
        
    def getHistoricalData(self, ticker_list, max_number_of_days = -1, start_date = None):
        '''Obtains the entirety of the historical data fore each of the stocks that is possible
        @param start_date: datetime.datetime object matching the final day in the period
            stock data should be obtained for. If this is None, the current day will be used
        @param max_number_of_days: number representing the maximum amount of days BEFORE start_date
            stock data should be obtained for. I.E. a value of 5 will result in stock data 
            being obtained for the period (start_date - 5 days, start_date) 
        '''
        errored = []
        data = []
        for ticker in ticker_list:
            ret = self.__getDataForTicker(ticker, max_number_of_days, start_date)
            if ret[0]:
                data.append([ticker, ret[1]])
            else:
                errored.extend([ticker])
        
        return [data, errored]
    
    def __getDataForTicker(self, ticker, max_number_of_days, start_date):
        '''Obtain historical data for the ticker
        @param ticker: Ticker to obtain data for
        @param max_number_of_days: maximum number of days before start_date to obtain data for
        @param start_date: The final day to obtain data for
        Obtains the historical data for the ticker, using the cookie manager to manage the HTTP request, and
        handling the URL construction required to fit into Yahoo!'s downloading scheme. 
        '''
        period2 = None
        period1 = None
        
        # Convert the value of start_date into the number of seconds since the Epoch (Jan 1, 1970 at midnight)
        if start_date == None:
            start_date = datetime.now()
            period2 = round(start_date.timestamp())
        else:
            period2 = round(start_date.timestamp())
        
        # Use start_date and max_number_of_days to calculate the lower bound of the data retrieval period
        # (in the number of seconds since the Epoch)
        if max_number_of_days == -1:
            period1 = 0
        else:
            period1 = round((start_date - timedelta(days = max_number_of_days)).timestamp())

        # Setup full download URL from the base
        download_url = self.__urlBase.format(ticker.upper(), period1, period2, self.cookie_man.getCrumb()[1:])
        logger.logger.log(logger.INFORMATION, download_url)
        stat = openURL(download_url, cookie=self.cookie_man.getCookie())
        
        data = []
        # stat[0] is a flag for whether the connection went through
        if stat[0]:
            reply = stat[1]
            for line in reply:
                data.append(line.decode())
            return [True, data]
        else:
            logger.logger.log(logger.WARNING, "Ticker {0} errored with HTTP code {1}".format(ticker, stat[1]))
            return [False]

