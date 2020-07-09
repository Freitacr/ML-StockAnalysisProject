'''
Created on Dec 22, 2017

@author: Colton Freitas, Jim Carey
'''


from ..stock_data_downloading.downloader_yahoo import DownloaderYahoo
from general_utils.mysql_management.mysql_tables import stock_list_table
from general_utils.mysql_management.mysql_tables import stock_data_table
from datetime import datetime as dt, timedelta
from mysql.connector.errors import ProgrammingError
from general_utils.logging import logger


class YahooDataFormatting:
    
    def __init__(self, ticker_list):
        '''Initialization method
        @param ticker_list: List of stock tickers to obtain and format data for
        @param login_credentials: Login credentials for the MySQL Server 
        '''
        self.ticker_list = ticker_list
        self.data_downloader = DownloaderYahoo()
        self.stock_list_table = stock_list_table.StockListTable()
        # for each ticker stored in stock_list, grab the ticker and whether yahoo data has been downloaded for it
        stored_tickers = self.stock_list_table.select_from_table(
            [stock_list_table.TICKER_COLUMN_NAME, stock_list_table.YAHOO_COLUMN_NAME]
        )
        down_days = self.generateDownloadDays(stored_tickers)
        
        self.data = self.obtainData(down_days)

    def obtainCurrentRecords(self, stock_ticker):
        '''Obtains historical data on the supplied ticker from the MYSQL Database
        @param stock_ticker: The ticker to obtain the data for
        '''
        column_list = ["hist_date", "adj_close"]
        data_table = stock_data_table.StockDataTable(f"{stock_ticker.upper}_yahoo_data")
        return data_table.select_from_table(column_list, conditional="order by hist_date")
    
    def generateDownloadDays(self, stored_tickers):
        '''Generates a list of days that data should be collected for
        @param stored_tickers: List of tickers that download days should be generated for 
        '''
        download_days = []
        one_day_change = timedelta(days = 1)
        
        for ticker in self.ticker_list:
            #Check whether ticker is already stored in the database
            stored = False
            for table_entry in stored_tickers:
                #if table_entry is the correct ticker and is in the database
                if table_entry[0].lower() == ticker.lower() and table_entry[1]:
                    stored = True
                    break
            
            if not stored:
                download_days.extend([[ticker.upper(), 'all']])    
                continue
            
            
            stored_days = None
            try:
                stored_days = self.obtainCurrentRecords(ticker)
            except ProgrammingError as e:
                logger.logger.log(logger.NON_FATAL_ERROR, "%s errored: %s" % (ticker, str(e)))
            
            req_days = []
            
            # Check whether the table that should house the data for the current ticker is empty or non-existant
            # if so, then all data should be accepted
            if stored_days == None:
                download_days.extend([[ticker.upper(), 'all']])
                continue
            else:
                start_date = stored_days[0][0]
                # If there's a place in the adj_close column (stored_days[:][1]) that has -1,
                # It needs to be updated. This just checks the first day for it.
                if stored_days[0][1] == -1:
                    req_days.extend([start_date])
            # For each day after the start_date in the stored days, check for missing days
            # Any days that are missing, added them to the download_days list
            for stored_day in stored_days[1:]:
                start_date = start_date + one_day_change
                if not stored_day[0] == start_date:
                    req_days.extend([start_date])
                    while not stored_day[0] == start_date:
                        start_date = start_date + one_day_change
                        if not stored_day[0] == start_date:
                            req_days.extend([start_date])
            
            today = dt.date(dt.now())
            
            # Since the final stored day may not be the current date, add each day between the final stored day and
            # today the download_days list
            if not stored_days [-1][0] == today:
                while start_date < today:
                    start_date += one_day_change
                    req_days.extend([start_date])
            download_days.extend([[ticker.upper(), req_days]])
            
        return download_days
    
    def obtainData(self, down_days):
        '''Attempts to obtain data for all tickers in self.ticker_list
        @param down_days: List of days to keep data for with respect to each ticker
        Attempts to download data from the ticker list, repeating any errored tickers up to two more times to get data
        Then uses the down_days list to filter out unneeded data, returning data in the following format
        @return: ['yahoo', [ [ticker, [day1data, day2data... dayNdata]] , [ticker2, ...] ... [tickerN, ...] ] ] 
        '''
        ret = []
        
        # Extraction of the actual ticker from the down_days list structure
        download_tickers = []
        for ticker in down_days:
            download_tickers.extend([ticker[0]])

        data = []
        temp_data, errored = self.data_downloader.getHistoricalData(download_tickers)
        data.extend(temp_data)
        temp_data, errored = self.data_downloader.getHistoricalData(errored)
        data.extend(temp_data)
        temp_data, errored = self.data_downloader.getHistoricalData(errored)
        data.extend(temp_data)

        for data_ticker in data:
            logger.logger.log(logger.INFORMATION, "Now formatting data for %s" % data_ticker[0])
            # Grab the download days for the current ticker for reference
            download_days = None
            for down_ticker in down_days:
                if down_ticker[0] == data_ticker[0]:
                    download_days = down_ticker[1]

            if not download_days == 'all':
                for index in range(len(download_days)):
                    download_days[index] = download_days[index].isoformat()
                
                ticker_ret = [x for x in data_ticker[1] if x[0].split(",")[0] in download_days]
            elif download_days == 'all':
                ticker_ret = data_ticker[1][1:]
            else:
                ticker_ret = None
            
            ticker_ret = [data_ticker[0], ticker_ret]
            ret.extend([ticker_ret])
        ret = ['yahoo', ret]
        return ret
    
    def getData(self):
        '''Getter method for self.data'''
        return self.data
