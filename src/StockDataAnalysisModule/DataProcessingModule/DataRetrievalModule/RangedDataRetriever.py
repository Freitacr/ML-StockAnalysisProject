from GeneralUtils.StockDataMYSQLManagement.MSYQLDataManipulator import MYSQLDataManipulator

class RangedDataRetriever:

    def __init__ (self, login_credentials, column_list, start_date: str, end_date: str):
        self.data_man = MYSQLDataManipulator(login_credentials[0], login_credentials[1], login_credentials[2], login_credentials[3])
        self.column_list = column_list
        self.__setupStockSourceList()
        self.startDate = start_date
        self.endDate = end_date

    def __setupStockSourceList(self):
        # Right now this is going to be a lot more temporary, just using whatever source has actual data on the stock
        # The sources are also going to be in order of preference of use
        # (Yahoo, then Google right now. Even though Google has no data
        # Downloading yet)
        source_availability = self.data_man.select_from_table('stock_list', ['ticker', 'yahoo', 'google'])
        tickers = [x[0] for x in source_availability]
        yahoo_availability = [x[1] for x in source_availability]
        self.data_sources = {}
        for index in range(len(tickers)):
            self.data_sources[tickers[index]] = []
            if yahoo_availability[index]:
                self.data_sources[tickers[index]].append('yahoo')
            else:
                raise ValueError("%s has no data, but exists in the database. This is bad." % tickers[index])
    
    def close(self):
        self.data_man.close(commit=False)

    def retrieveData(self, ticker, source='yahoo'):
        if ticker not in self.data_sources:
            raise KeyError("%s did not have data in the database" % ticker)
        table = "%s_%s_data" % (ticker, source)
        dateConditional = 'where hist_date > date(\"%s\") and hist_date < date(\"%s\")' % (self.startDate, self.endDate)
        return self.data_man.select_from_table(table, self.column_list, conditional=dateConditional)