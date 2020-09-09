'''
Created on Dec 19, 2017

@author: Colton Freitas
'''
from datetime import datetime as dt
import importlib
import os
import sys

from general_utils.mysql_management.mysql_tables import stock_data_table, stock_list_table
from general_utils.logging import logger
from stock_data_downloading_module.stock_data_formatting import data_formatting


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
    data_split = day_data.rstrip().split(",")
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

    providers = os.listdir("stock_data_downloading_module/stock_data_formatters")
    for provider in providers:
        if provider.startswith('__'):
            continue
        importlib.import_module('stock_data_downloading_module.stock_data_formatters.' + provider.replace('.py', ''))

    # This date is exclusive.
    args = sys.argv[1:]
    if args:
        final_date = args[0]
        final_date = dt.strptime(final_date, "%d/%m/%y")
    else:
        final_date = None

    data_formatter = data_formatting.registry
    data = data_formatter.get_data(stock_list, final_date)

    for data_source in data:
        source_string, source_data = data_source
        for ticker_formatted_data in source_data:
            stock_ticker = ticker_formatted_data.ticker
            stock_data = ticker_formatted_data.data
            stock_status = stock_list_db_table.select_from_table(
                [source_string],
                conditional=f"where ticker = '{stock_ticker}'"
            )
            ticker_data_table = stock_data_table.StockDataTable(f"{stock_ticker}_{source_string}_data")
            for i in range(len(stock_data)):
                stock_data[i] = convert_upload_data(stock_data[i])
            ticker_data_table.insert_into_table(
                stock_data,
                [
                    stock_data_table.HISTORICAL_DATE_COLUMN_NAME,
                    stock_data_table.HIGH_PRICE_COLUMN_NAME,
                    stock_data_table.LOW_PRICE_COLUMN_NAME,
                    stock_data_table.OPEN_PRICE_COLUMN_NAME,
                    stock_data_table.CLOSING_PRICE_COLUMN_NAME,
                    stock_data_table.ADJUSTED_CLOSING_PRICE_COLUMN_NAME,
                    stock_data_table.VOLUME_COLUMN_NAME
                ])
            if len(stock_status) == 0:
                stock_list_db_table.insert_into_table(
                    [(stock_ticker, True)],
                    [
                        stock_list_table.TICKER_COLUMN_NAME,
                        source_string
                    ])
            elif not stock_status[0][0]:
                # meaning that the source was not listed as being available for the current stock
                ticker_data_table.update(f"set {source_string}=1", f"where ticker='{stock_ticker}'")

    logger.logger.log(logger.INFORMATION, "Finished obtaining data from data sources")
