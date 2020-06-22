import sys
import os
import importlib
from DataProvidingModule.DataProviderRegistry import registry
from configparser import ConfigParser, NoOptionError, NoSectionError

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"


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


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    login_credentials = config_handling()
    providers = os.listdir("DataProvidingModule/DataProviders")
    for provider in providers:
        if provider.startswith('__'):
            continue
        importlib.import_module('DataProvidingModule.DataProviders.' + provider.replace('.py', ''))
    consumers = os.listdir("TrainingManagers")
    for consumer in consumers:
        if consumer.startswith('__'):
            continue
        importlib.import_module("TrainingManagers." + consumer.replace('.py',''))
    registry.passData(login_credentials, args[0], stop_for_errors=True)
