'''
Created on Dec 24, 2017

@author: Colton Freitas
'''

from configparser import ConfigParser, NoSectionError, NoOptionError
from StockDataAnalysisModule.MLManager import MLManager

def write_default_configs(parser, file_position):
    parser.add_section('login_credentials')
    parser.set('login_credentials', 'user', 'root')
    parser.set('login_credentials', 'password', "")
    parser.set('login_credentials', 'database', 'stock_testing')
    parser.set('login_credentials', 'host', 'localhost')
    fp = open(file_position, 'w')
    parser.write(fp)
    fp.close()

def config_handling():
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




if __name__ == "__main__":
    login_credentials = config_handling()
    ml_manager = MLManager(login_credentials)
    ml_manager.basicMovementsTraining("../model_data", ml_model = 'decision_tree')
    print("Finished")







