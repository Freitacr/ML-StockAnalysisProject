from configparser import ConfigParser, NoSectionError, NoOptionError
from StockDataAnalysisModule.DataProcessingModule.StockClusterDataManager import StockClusterDataManager
import datetime as dt


def write_default_configs(parser, file_position):
    '''Writes the default configuration file at file_position'''
    parser.add_section('login_credentials')
    parser.set('login_credentials', 'user', 'root')
    parser.set('login_credentials', 'password', "")
    parser.set('login_credentials', 'database', 'stock_testing')
    parser.set('login_credentials', 'host', 'localhost')
    fp = open(file_position, 'w')
    parser.write(fp)
    fp.close()


def config_handling():
    '''Handles reading in and error checking the configuration data'''
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


if __name__ == '__main__':
    exit(0)
    login_credentials = config_handling()
    import sys
    args = sys.argv[1:]
    startDate = None
    endDate = None
    try:
        startDate = args[0]
        endDate = args[1]
    except IndexError:
        pass
    clusterManager = StockClusterDataManager(login_credentials, startDate, endDate)
    x_train, y_train, x_test, y_test = clusterManager.retrieveNormalizedTrainingDataMovementTargets()
    print()