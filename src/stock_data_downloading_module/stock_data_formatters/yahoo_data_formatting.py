'''
Created on Dec 22, 2017

@author: Colton Freitas, Jim Carey
'''


import datetime
from typing import List, Tuple, Dict, Set

from mysql.connector.errors import ProgrammingError

from stock_data_downloading_module.stock_data_downloading.downloader_yahoo import DownloaderYahoo
from general_utils.mysql_management.mysql_tables import stock_list_table
from general_utils.mysql_management.mysql_tables import stock_data_table
from general_utils.logging import logger
from stock_data_downloading_module.stock_data_formatting import data_formatting


def _obtain_current_records(stock_ticker) -> List[Tuple[datetime.date, float]]:
    '''Obtains historical data on the supplied ticker from the MYSQL Database
            @param stock_ticker: The ticker to obtain the data for
            '''
    column_list = ["hist_date", "adj_close"]
    data_table = stock_data_table.StockDataTable(f"{stock_ticker.upper()}_yahoo_data")
    return data_table.select_from_table(column_list, conditional="order by hist_date")


def _convert_date_str_to_timestamp(date_str: str):
    return datetime.datetime.strptime(date_str, "%Y-%m-%d").timestamp()


def _generate_stored_day_timestamps(stored_tickers: Dict[str, bool], ticker_list: List[str]) -> Dict[str, Set]:
    '''Generates a list of days that data should be collected for
    @param stored_tickers: List of tickers that download days should be generated for
    '''
    ret_days = {}

    for ticker in ticker_list:
        if ticker not in stored_tickers or not stored_tickers[ticker]:
            ret_days[ticker.upper()] = set()
            continue

        stored_days = None
        try:
            stored_days = _obtain_current_records(ticker)
        except ProgrammingError as e:
            logger.logger.log(logger.NON_FATAL_ERROR, "%s errored: %s" % (ticker, str(e)))

        swap_set = set()
        for stored_day in stored_days:
            stored_date, adj_close = stored_day
            if not adj_close == -1:
                stored_datetime = datetime.datetime(
                    year=stored_date.year,
                    month=stored_date.month,
                    day=stored_date.day
                )
                swap_set.add(stored_datetime.timestamp())
        stored_days = swap_set
        ret_days[ticker.upper()] = stored_days

    return ret_days


class YahooDataFormatting(data_formatting.DataFormatter):
    
    def __init__(self):
        self.data_downloader = None
        self.stock_list_table = None
        data_formatting.registry.formatter_registry.add(self)

    def _obtain_data(self, already_stored_dates):
        '''Attempts to obtain data for all tickers in self.ticker_list
        @param already_stored_dates: Dictionary mapping the upper cased ticker to a set of timestamps that already
            have data in the database.
        Attempts to download data from the ticker list, repeating any errored tickers up to two more times to get data
        Then uses the down_days list to filter models unneeded data, returning data in the following format
        @return: ['yahoo', [ [ticker, [day1data, day2data... dayNdata]] , [ticker2, ...] ... [tickerN, ...] ] ] 
        '''
        ret = []
        
        # Extraction of the actual ticker from the down_days list structure
        download_tickers = []
        for ticker in already_stored_dates:
            download_tickers.append(ticker)

        data = []
        temp_data, errored = self.data_downloader.getHistoricalData(download_tickers)
        data.extend(temp_data)
        temp_data, errored = self.data_downloader.getHistoricalData(errored)
        data.extend(temp_data)
        temp_data, errored = self.data_downloader.getHistoricalData(errored)
        data.extend(temp_data)

        for data_ticker in data:
            ticker, data = data_ticker
            logger.logger.log(logger.INFORMATION, f"Now formatting data for {ticker}")
            dates_stored = already_stored_dates[ticker.upper()]
            data = data[1:]  # ignore the line telling us which column is which.
            ticker_ret = []
            for day_data in data:
                hist_date_string = day_data.split(",")[0]
                hist_timestamp = _convert_date_str_to_timestamp(hist_date_string)
                if hist_timestamp not in dates_stored:
                    ticker_ret.append(day_data)

            ticker_ret = data_formatting.TickerFormattedData(data_ticker[0], ticker_ret)
            ret.append(ticker_ret)
        return ret
    
    def get_data(self, ticker_list: List[str]) -> Tuple[str, List[data_formatting.TickerFormattedData]]:
        # Initialization moved here as class may be instanced and not used for a current download
        self.data_downloader = DownloaderYahoo() if self.data_downloader is None else self.data_downloader
        self.stock_list_table = stock_list_table.StockListTable() \
            if self.stock_list_table is None else self.stock_list_table

        stored_tickers = self.stock_list_table.select_from_table(
            [stock_list_table.TICKER_COLUMN_NAME, stock_list_table.YAHOO_COLUMN_NAME]
        )
        stored_ticker_dict = {}
        for ticker, stored in stored_tickers:
            stored_ticker_dict[ticker] = stored
        down_days = _generate_stored_day_timestamps(stored_ticker_dict, ticker_list)
        return 'yahoo', self._obtain_data(down_days)


formatter = YahooDataFormatting()
