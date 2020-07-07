from configparser import ConfigParser, NoOptionError, NoSectionError
from os.path import exists


def write_default_configs(file_position):
    parser.add_section('login_credentials')
    parser.add_section('execution_options')
    parser.set('login_credentials', 'user', 'stock_worker')
    parser.set('login_credentials', 'database', 'stock_testing')
    parser.set('login_credentials', 'host', 'localhost')
    parser.set("execution_options", "predict", "False")
    fp = open(file_position, 'w')
    parser.write(fp)
    fp.close()


def read_login_credentials():
    user = parser.get("login_credentials", 'user')
    database = parser.get('login_credentials', 'database')
    host = parser.get('login_credentials', 'host')
    return host, user, database


def read_execution_options():
    prediction = parser.getboolean("execution_options", "predict")
    return prediction


def read_config_or_write_defaults():
    if not exists(config_filepath):
        write_default_configs(config_filepath)
    try:
        with open(config_filepath, 'r') as fp:
            parser.read_file(fp)
        read_login_credentials()
    except (NoSectionError, NoOptionError) as err:
        print(err)
        write_default_configs(config_filepath)


def update_config():
    with open(config_filepath, 'w') as fp:
        parser.write(fp)


config_filepath = "../configuration/config.ini"

try:
    parser: "ConfigParser" = parser
except NameError:
    parser = ConfigParser()
    read_config_or_write_defaults()


