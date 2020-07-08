'''
Created on Dec 22, 2017

@author: Colton Freitas, Jim Carey
'''


from ..stock_data_downloading.downloader_yahoo import DownloaderYahoo
from general_utils.mysql_management.mysql_data_manipulator import MYSQLDataManipulator
from datetime import datetime as dt, timedelta
from mysql.connector.errors import ProgrammingError
from general_utils.logging import logger


class YahooDataFormatting:
    
    def __init__(self, ticker_list, login_credentials):
        '''Initialization method
        @param ticker_list: List of stock tickers to obtain and format data for
        @param login_credentials: Login credentials for the MySQL Server 
        '''
        self.login_credentials = login_credentials
        self.ticker_list = ticker_list
        self.__setupYahManager()
        self.data_downloader = DownloaderYahoo()
        
        #for each ticker stored in stock_list, grab the ticker and the boolean of whether yahoo data has been downloaded for it
        stored_tickers = self.yah_manager.select_from_table('stock_list', ['ticker', 'yahoo'])
        down_days = self.generateDownloadDays(stored_tickers)
        
        self.data = self.obtainData(down_days)
        
        self.yah_manager.close()
    
    def __setupYahManager(self):
        '''Sets up the MYSQL manipulator for ease of querying the server'''
        host = self.login_credentials[0]
        user = self.login_credentials[1]
        password = self.login_credentials[2]
        database = self.login_credentials[3]
        self.yah_manager = MYSQLDataManipulator(host, user, password, database)
    
    
    def obtainCurrentRecords(self, stock_ticker):
        '''Obtains historical data on the supplied ticker from the MYSQL Database
        @param stock_ticker: The ticker to obtain the data for
        '''
        column_list = ["hist_date", "adj_close"]
        return self.yah_manager.select_from_table("%s_yahoo_data" % stock_ticker.upper(), column_list, conditional = 'order by hist_date')
    
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
        
        # This allows for several attempts to be made at getting data for a ticker (hopefully to avoid events where internet connectivity is spotty)
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
            
            # Convert download days list from containing datetime objects into iso formatted strings (AKA the same string Yahoo uses for dates)
            if not download_days == 'all':
                for index in range(len(download_days)):
                    download_days[index] = download_days[index].isoformat()
            
                # what follows is code to make the weekends and weekdays that are missed a part of the dataset using special markings
                # ticker_ret = []
                # for x in data_ticker[1]:
                #    for d_day in download_days:
                #        if x[0].split(",")[0] == d_day:
                #            download_days.remove(d_day)
                #            ticker_ret.extend(x)
                #            break
                # Do custom filling for days that are weirdly missed.
                # for d_day in download_days:
                #    replace_string = "%s,%s,%s,%s,%s,%s,%s\n"
                #    day = dt.strptime(d_day, "%Y-%m-%d")
                #    if day.weekday() >= 5:
                #        replace_string = replace_string % (d_day, "-", "-", "-", "-", "-", "-")
                #    else:
                #        replace_string = replace_string % (d_day, "!", "!", "!", "!", "!", "!")
                #    ticker_ret.extend([replace_string])
                # This is commented out since it is still up in the air about whether it should be used or not. It works, however.
                
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
