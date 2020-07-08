from typing import List, Optional, Iterable, Any

from general_utils.mysql_management import mysql_data_manipulator


def construct_drop_table_func(table_name: str):
    def drop_table_func(data_man: mysql_data_manipulator.MYSQLDataManipulator):
        data_man.drop_table(table_name)
    return drop_table_func


def _check_column_existence(table_name: str, column_list: List[str], column_names: List[str]):
    for col in column_list:
        if col not in column_names:
            raise ValueError("Column %s was requested from table %s, but the table does not contain a column"
                             "with that identifier." % (col, table_name))


def construct_select_from_table_func(table_name: str, column_names: List[str]):
    def select_from_table(
            data_man: mysql_data_manipulator.MYSQLDataManipulator,
            column_list: List[str],
            database: Optional[str] = None,
            conditional: Optional[str] = None):
        _check_column_existence(table_name, column_list, column_names)
        return data_man.select_from_table(table_name, column_list, database, conditional)
    return select_from_table


def construct_insert_into_table_func(table_name: str, column_names: List[str]):
    def insert_into_table(
            data_man: mysql_data_manipulator.MYSQLDataManipulator,
            data: List[Iterable[Any]],
            column_list: Optional[List[str]] = None,
            database: Optional[str] = None):
        cols = column_names
        if column_list is not None:
            _check_column_existence(table_name, column_list, column_names)
            cols = column_list
        data_man.insert_into_table(table_name, cols, data, database)
    return insert_into_table
