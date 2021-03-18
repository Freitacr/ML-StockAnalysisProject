from general_utils.mysql_management.mysql_tables import stock_list_table
from general_utils.mysql_management.mysql_tables import stock_data_table


class RangedDataRetriever:

    def __init__ (self, column_list, start_date: str, end_date: str):
        self.column_list = column_list
        self.__setupStockSourceList()
        self.startDate = start_date
        self.endDate = end_date

    def __setupStockSourceList(self):
        # Right now this is going to be a lot more temporary, just using whatever source has actual data on the stock
        # The sources are also going to be in order of preference of use
        # (Yahoo, then Google right now. Even though Google has no data
        # Downloading yet)
        source_availability_table = stock_list_table.StockListTable()
        source_availability = source_availability_table.select_from_table(['ticker', 'yahoo', 'google'])
        tickers = [x[0] for x in source_availability]
        yahoo_availability = [x[1] for x in source_availability]
        self.data_sources = {}
        for index in range(len(tickers)):
            self.data_sources[tickers[index]] = []
            if yahoo_availability[index]:
                self.data_sources[tickers[index]].append('yahoo')
            else:
                raise ValueError("%s has no data, but exists in the database. This is bad." % tickers[index])

    def retrieveData(self, ticker, source='yahoo'):
        if ticker not in self.data_sources:
            raise KeyError("%s did not have data in the database" % ticker)
        table = stock_data_table.StockDataTable(f"{ticker}_{source}_data")
        date_conditional = f'where hist_date > date("{self.startDate}") and hist_date < date("{self.endDate}")'
        return table.select_from_table(self.column_list, conditional=date_conditional)
