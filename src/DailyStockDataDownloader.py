'''
Created on Dec 19, 2017

@author: Colton Freitas
'''
from GeneralUtils.StockDataMYSQLManagement.MSYQLDataManipulator import MYSQLDataManipulator
from datetime import datetime as dt
from StockDataDownloadingModule.StockDataFormatting.DataFormatting import DataFormatter
from configparser import ConfigParser, NoSectionError, NoOptionError

def write_default_configs(parser, file_position):
    '''Creates the default configuration file in file_position with default values'''
    
    parser.add_section('login_credentials')
    parser.set('login_credentials', 'user', 'root')
    parser.set('login_credentials', 'password', "")
    parser.set('login_credentials', 'database', 'stock_testing')
    parser.set('login_credentials', 'host', 'localhost')
    fp = open(file_position, 'w')
    parser.write(fp)
    fp.close()

def config_handling():
    '''does all of the configuration handling using the configparser package'''
    file_position = "../configuration_data/config.ini"
    parser = ConfigParser()
    try:
        fp = open(file_position, 'r')
        fp.close()
    except FileNotFoundError:
        write_default_configs(parser, file_position)
    config_file = open(file_position, 'r')
    parser.read_file(config_file)
    try:
        user = parser.get('login_credentials', 'user')
        password = parser.get('login_credentials', 'password')
        database = parser.get('login_credentials', 'database')
        host = parser.get('login_credentials', 'host')
    except (NoSectionError, NoOptionError):
        write_default_configs(parser, file_position)
        user = parser.get('login_credentials', 'user')
        password = parser.get('login_credentials', 'password')
        database = parser.get('login_credentials', 'database')
        host = parser.get('login_credentials', 'host')
    return [host, user, password, database]


def convertAndInsertData(day_data, source_string, stock_ticker):
    '''Converts all data into the correct format and uploads it to the MYSQL table '''
    data_string = day_data[0]
    data_split = data_string.rstrip().split(",")
    day = dt.strptime(data_split[0], "%Y-%m-%d")
    try:
        open_price = float(data_split[1])
        high_price = float(data_split[2])
        low_price = float(data_split[3])
        close_price = float(data_split[4])
        adj_close = float(data_split[5])
        volume_data = int (data_split[6])
    except ValueError:
        for data_index in range(len(data_split[1:])):
            if data_split[data_index + 1] == 'null':
                data_split[data_index + 1] = -1
        open_price = float(data_split[1])
        high_price = float(data_split[2])
        low_price = float(data_split[3])
        close_price = float(data_split[4])
        adj_close = float(data_split[5])
        volume_data = int (data_split[6])
    upload_data = [day, open_price, high_price, low_price, close_price, adj_close, volume_data]
    
    data_manager.insert_into_table("%s_%s_data" % (stock_ticker, source_string), col_list, [upload_data])
    
def get_stock_list():
    '''Obtains a list of all stock tickers to attempt to download'''
    file = open("../configuration_data/stock_list.txt", 'r')
    return_data = []
    for line in file:
        return_data.extend([line.strip()])
    file.close()
    return return_data
    
                    
if __name__ == '__main__':
    #lists of what types of data are stored... TODO: Refine these two to be a bit... better...
    col_list = ["hist_date", "high_price", "low_price", "opening_price", "close_price", "adj_close", "volume_data"]
    creation_col_list = [["id int primary key auto_increment"], [col_list[0], "Date"], [col_list[1], "float"],
                         [col_list[2], "float"], [col_list[3], "float"], [col_list[4], "float"], [col_list[5], "float"],
                         [col_list[6], "long"]]
    
    
    login_credentials = config_handling()
    stock_list = get_stock_list()
    
    data_formatter = DataFormatter(login_credentials, stock_list)
    data = data_formatter.getData()
    data_manager = MYSQLDataManipulator(login_credentials[0], login_credentials[1], login_credentials[2], login_credentials[3])
    
    #data[0][1] is the actual data storage. It contains a list for each of the tickers data is returned for
    #data[0][1][0] is the index of the first ticker's data list. index 0 is the ticker, 1 is the actual data list  
    #data[0][1][0][1] is the data list, it contains lists (each of size 1...) containing each of the days of data
    
    
    for data_source in data:
        source_string = data_source[0]
        for ticker_data in data_source[1]:
            stock_ticker = (ticker_data[0])
            stock_status = data_manager.select_from_table("stock_list", [source_string], conditional = 'where ticker = \'%s\'' % (stock_ticker))
            if len(stock_status) == 0 or not stock_status[0][0]:
                #create new table and upload data
                if not len(stock_status) == 0:
                    #meaning that the stock_status is 0
                    data_manager.create_table("%s_%s_data" % (stock_ticker, source_string), creation_col_list)
                    for day_data in ticker_data[1]:
                        convertAndInsertData(day_data, source_string, stock_ticker)
                    data_manager.execute_sql( ("update stock_list") + (" set %s=1" % source_string) + (' where ticker = \'%s\'' % (stock_ticker)))
                else:
                    data_manager.create_table("%s_%s_data" % (stock_ticker, source_string), creation_col_list)
                    for day_data in ticker_data[1]:
                        convertAndInsertData(day_data, source_string, stock_ticker)
                    data_manager.insert_into_table("stock_list", ['ticker', source_string], [[stock_ticker, True]])
            elif stock_status[0][0]:
                for day_data in ticker_data[1]:
                    convertAndInsertData(day_data, source_string, stock_ticker)
    data_manager.close(commit = True)
    