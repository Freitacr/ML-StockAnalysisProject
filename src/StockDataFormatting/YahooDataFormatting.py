'''
Created on Dec 22, 2017

@author: Colton Freitas
'''

from StockDataDownloading.DownloaderYahoo import DownloaderYahoo
from StockDataMYSQLManagement.MSYQLDataManipulator import MYSQLDataManipulator
from datetime import datetime as dt, timedelta
from mysql.connector.errors import ProgrammingError

class YahooDataFormatting:
    
    def __init__(self, ticker_list, login_credentials):
        self.login_credentials = login_credentials
        self.ticker_list = ticker_list
        self.__setupYahManager()
        self.data_downloader = DownloaderYahoo()
        
        stored_tickers = self.yah_manager.select_from_table('stock_list', ['ticker', 'yahoo'])
        down_days = self.generateDownloadDays(stored_tickers)
        
        self.data = self.obtainData(down_days)
        
        self.yah_manager.close()
    
    def __setupYahManager(self):
        host = self.login_credentials[0]
        user = self.login_credentials[1]
        password = self.login_credentials[2]
        database = self.login_credentials[3]
        self.yah_manager = MYSQLDataManipulator(host, user, password, database)
    
    
    def obtainCurrentRecords(self, stock_ticker):
        column_list = ["hist_date", "adj_close"]
        return self.yah_manager.select_from_table("%s_yahoo_data" % stock_ticker.upper(), column_list, conditional = 'order by hist_date')
    
    def obtainData(self, down_days):
        ret = []
        
        download_tickers = []
        for ticker in down_days:
            download_tickers.extend([ticker[0]])
        
        #This allows for several attempts to be made at getting data for a ticker (hopefully to avoid events where internet connectivity is spotty)
        
        data = []
        temp_data, errored = self.data_downloader.getHistoricalData(download_tickers)
        data.extend(temp_data)
        temp_data, errored = self.data_downloader.getHistoricalData(errored)
        data.extend(temp_data)
        temp_data, errored = self.data_downloader.getHistoricalData(errored)
        data.extend(temp_data)
        
        for data_ticker in data:
            download_days = None
            for down_ticker in down_days:
                if down_ticker[0] == data_ticker[0]:
                    download_days = down_ticker[1]
            #Convert download days list from containing datetime objects into iso formatted strings (AKA the same string Yahoo uses for dates)
            if not download_days == 'all':
                for index in range(len(download_days)):
                    download_days[index] = download_days[index].isoformat()
                
                ticker_ret = []
                for x in data_ticker[1]:
                    for d_day in download_days:
                        if x[0].split(",")[0] == d_day:
                            download_days.remove(d_day)
                            ticker_ret.extend(x)
                            break
                #Do custom filling for days that are weirdly missed. 
                #for d_day in download_days:
                #    replace_string = "%s,%s,%s,%s,%s,%s,%s\n"
                #    day = dt.strptime(d_day, "%Y-%m-%d")
                #    if day.weekday() >= 5:
                #        replace_string = replace_string % (d_day, "-", "-", "-", "-", "-", "-")
                #    else:
                #        replace_string = replace_string % (d_day, "!", "!", "!", "!", "!", "!")
                    
                    
                #    ticker_ret.extend([replace_string])
                ticker_ret = [x for x in data_ticker[1] if x[0].split(",")[0] in download_days]
            elif download_days == 'all':
                ticker_ret = data_ticker[1]
            else:
                ticker_ret = None
            
            ticker_ret = [data_ticker[0], ticker_ret]
            ret.extend([ticker_ret])
        ret = ['yahoo', ret]
        return ret
    
    def getData(self):
        return self.data
    
    def generateDownloadDays(self, stored_tickers):
        download_days = []
        one_day_change = timedelta(days = 1)
        for ticker in self.ticker_list:
            stored = False
            for table_entry in stored_tickers:
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
                print(e)
            
            req_days = []
            
            if stored_days == None:
                download_days.extend([[ticker.upper(), 'all']])
            else:
                start_date = stored_days[0][0]
                if stored_days[0][1] == -1:
                    req_days.extend([start_date])
            
            for stored_day in stored_days[1:]:
                start_date = start_date + one_day_change
                if not stored_day[0] == start_date:
                    req_days.extend([start_date])
                    while not stored_day[0] == start_date:
                        start_date = start_date + one_day_change
                        if not stored_day[0] == start_date:
                            req_days.extend([start_date])
            
            today = dt.date(dt.now())
            
            if not stored_days [-1][0] == today:
                while start_date < today:
                    start_date += one_day_change
                    req_days.extend([start_date])
            download_days.extend([[ticker.upper(), req_days]])
        return download_days
    
    
    
    
    
    
    
    
    
    
    