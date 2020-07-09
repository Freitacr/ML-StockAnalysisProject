'''
Created on Dec 19, 2017

@author: Colton Freitas
'''
from datetime import datetime as dt

from general_utils.mysql_management.mysql_tables import stock_data_table, stock_list_table
from general_utils.logging import logger
from stock_data_downloading_module.stock_data_formatting.data_formatting import DataFormatter


def convertString(in_str: str, flag: str = 'float'):
    if in_str is None:
        return None
    if flag == 'float':
        if not in_str.replace('.', '').isnumeric():
            ret = -1
        else:
            ret = float(in_str)
    elif flag == 'int':
        if not in_str.isdigit():
            ret = -1
        else:
            ret = int(in_str)
    else:
        raise ValueError(f"Unsupported flag: {flag}")
    
    return ret


def convert_upload_data(day_data):
    '''Converts all data into the correct format and uploads it to the MYSQL table '''
    data_string = day_data[0]
    data_split = data_string.rstrip().split(",")
    day = dt.strptime(data_split[0], "%Y-%m-%d")
    open_price = convertString(data_split[1])
    high_price = convertString(data_split[2])
    low_price = convertString(data_split[3])
    close_price = convertString(data_split[4])
    adj_close = convertString(data_split[5])
    volume_data = convertString(data_split[6], flag='int')
    upload_data = [day, high_price, low_price, open_price, close_price, adj_close, volume_data]
    return upload_data


def get_stock_list():
    '''Obtains a list of all stock tickers to attempt to download'''
    file = open("../configuration/stock_list.txt", 'r')
    return_data = []
    for line in file:
        return_data.append(line.strip())
    file.close()
    return return_data
    
                    
if __name__ == '__main__':
    stock_list = get_stock_list()

    stock_list_db_table = stock_list_table.StockListTable()

    data_formatter = DataFormatter(stock_list)
    data = data_formatter.getData()

    # data[0][1] is the actual data storage. It contains a list for each of the tickers data is returned for
    # data[0][1][0] is the index of the first ticker's data list. index 0 is the ticker, 1 is the actual data list
    # data[0][1][0][1] is the data list, it contains lists (each of size 1...) containing each of the days of data

    for data_source in data:
        # TODO extract data source into its own class when it is created after all table interactions are abstracted.
        source_string = data_source[0]
        for ticker_data in data_source[1]:
            stock_ticker = (ticker_data[0])
            stock_status = stock_list_db_table.select_from_table(
                [source_string],
                conditional="where ticker = '%s'" % stock_ticker
            )
            ticker_data_table = stock_data_table.StockDataTable("%s_%s_data" % (stock_ticker, source_string))
            for i in range(len(ticker_data[1])):
                ticker_data[1][i] = convert_upload_data(ticker_data[1][i])
            ticker_data_table.insert_into_table(
                ticker_data[1],
                [
                    stock_data_table.HISTORICAL_DATE_COLUMN_NAME,
                    stock_data_table.HIGH_PRICE_COLUMN_NAME,
                    stock_data_table.LOW_PRICE_COLUMN_NAME,
                    stock_data_table.OPEN_PRICE_COLUMN_NAME,
                    stock_data_table.CLOSING_PRICE_COLUMN_NAME,
                    stock_data_table.ADJUSTED_CLOSING_PRICE_COLUMN_NAME,
                    stock_data_table.VOLUME_COLUMN_NAME
                ])
            if len(stock_status) == 0 or not stock_status[0][0]:
                if not len(stock_status) == 0:
                    # meaning that the source was not listed as being available
                    ticker_data_table.update(f"set {source_string}=1", f"where ticker='{stock_ticker}'")
                else:
                    stock_list_db_table.insert_into_table(
                        [(stock_ticker, True)],
                        [
                            stock_list_table.TICKER_COLUMN_NAME,
                            source_string
                        ])
    logger.logger.log(logger.INFORMATION, "Finished obtaining data from data sources")
