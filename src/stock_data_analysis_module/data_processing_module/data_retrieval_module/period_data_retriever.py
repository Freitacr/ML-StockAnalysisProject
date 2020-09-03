from general_utils.mysql_management.mysql_tables import stock_list_table
from general_utils.mysql_management.mysql_tables import stock_data_table


class PeriodDataRetriever:

    def __init__(self, column_list, end_date: str):
        self.column_list = column_list
        self.__setup_source_list()
        self.end_date = end_date

    def __setup_source_list(self):
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

    def retrieve_data(self, ticker, max_rows=-1, preferred_source='yahoo'):
        if ticker not in self.data_sources:
            raise KeyError("%s did not have data in the database" % ticker)
        source = preferred_source
        if preferred_source not in self.data_sources[ticker]:
            source = self.data_sources[ticker][0]
        table = stock_data_table.StockDataTable(f"{ticker}_{source}_data")
        conditional = f'where hist_date < date("{self.end_date}")'
        if max_rows > 0:
            conditional += f' limit {max_rows}'
        return table.select_from_table(self.column_list, conditional=conditional)
