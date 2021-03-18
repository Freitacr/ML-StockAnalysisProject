from typing import List, Optional, Iterable, Any

from general_utils.mysql_management import mysql_data_manipulator
from general_utils.config import config_parser_singleton
import atexit

_host, _user, _database = config_parser_singleton.read_login_credentials()
AUTO_CLOSING_DATA_MANAGER = mysql_data_manipulator.MYSQLDataManipulator(_host, _user, "", _database)
atexit.register(AUTO_CLOSING_DATA_MANAGER.close)


def construct_drop_table_func(table_name: str):
    def drop_table_func():
        AUTO_CLOSING_DATA_MANAGER.drop_table(table_name)
    return drop_table_func


def _check_column_existence(table_name: str, column_list: List[str], column_names: List[str]):
    for col in column_list:
        if col not in column_names:
            raise ValueError("Column %s was requested from table %s, but the table does not contain a column"
                             "with that identifier." % (col, table_name))


def construct_select_from_table_func(table_name: str, column_names: List[str]):
    def select_from_table(
            column_list: List[str],
            database: Optional[str] = None,
            conditional: Optional[str] = None):
        _check_column_existence(table_name, column_list, column_names)
        return AUTO_CLOSING_DATA_MANAGER.select_from_table(table_name, column_list, database, conditional)
    return select_from_table


def construct_insert_into_table_func(table_name: str, column_names: List[str]):
    def insert_into_table(
            data: List[Iterable[Any]],
            column_list: Optional[List[str]] = None,
            database: Optional[str] = None):
        cols = column_names
        if column_list is not None:
            _check_column_existence(table_name, column_list, column_names)
            cols = column_list
        AUTO_CLOSING_DATA_MANAGER.insert_into_table(table_name, cols, data, database)
    return insert_into_table


def construct_update_func(table_name: str):
    def update(
            update_sql: str,
            conditional: str):
        AUTO_CLOSING_DATA_MANAGER.update(table_name, update_sql, conditional)
    return update
