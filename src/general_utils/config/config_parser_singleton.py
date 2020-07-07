from configparser import ConfigParser, NoOptionError, NoSectionError
from os.path import exists

EXECUTION_OPTIONS_SECTION_NAME = 'execution_options'
EXECUTION_OPTIONS_PREDICT_IDENTIFIER = 'predict'

LOGIN_CREDENTIALS_SECTION_NAME = 'login_credentials'
LOGIN_CREDENTIALS_USER_IDENTIFIER = 'user'
LOGIN_CREDENTIALS_DATABASE_IDENTIFIER = 'database'
LOGIN_CREDENTIALS_HOST_IDENTIFIER = 'host'


def write_default_configs(file_position):
    parser.add_section(LOGIN_CREDENTIALS_SECTION_NAME)
    parser.add_section(EXECUTION_OPTIONS_SECTION_NAME)
    parser.set(
        LOGIN_CREDENTIALS_SECTION_NAME,
        LOGIN_CREDENTIALS_USER_IDENTIFIER,
        'stock_worker'
    )
    parser.set(
        LOGIN_CREDENTIALS_SECTION_NAME,
        LOGIN_CREDENTIALS_DATABASE_IDENTIFIER,
        'stock_testing'
    )
    parser.set(
        LOGIN_CREDENTIALS_SECTION_NAME,
        LOGIN_CREDENTIALS_HOST_IDENTIFIER,
        'localhost'
    )
    parser.set(
        EXECUTION_OPTIONS_SECTION_NAME,
        EXECUTION_OPTIONS_PREDICT_IDENTIFIER,
        "False"
    )
    fp = open(file_position, 'w')
    parser.write(fp)
    fp.close()


def read_login_credentials():
    user = parser.get(
        LOGIN_CREDENTIALS_SECTION_NAME,
        LOGIN_CREDENTIALS_USER_IDENTIFIER
    )
    database = parser.get(
        LOGIN_CREDENTIALS_SECTION_NAME,
        LOGIN_CREDENTIALS_DATABASE_IDENTIFIER
    )
    host = parser.get(
        LOGIN_CREDENTIALS_SECTION_NAME,
        LOGIN_CREDENTIALS_HOST_IDENTIFIER
    )
    return host, user, database


def read_execution_options():
    prediction = parser.getboolean(
        EXECUTION_OPTIONS_SECTION_NAME,
        EXECUTION_OPTIONS_PREDICT_IDENTIFIER
    )
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


