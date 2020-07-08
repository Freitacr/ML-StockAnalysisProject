'''
Created on Dec 20, 2017

@author: Colton Freitas
'''

import mysql.connector as connector
from mysql.connector.errors import Error as SQLError
from typing import Optional


_SQL_SPECIAL_CHARACTERS = [
    '"', "'", '\\', '%', '_'
]


def escape_table_name(table_name: str):
    ret_name = table_name
    for special_char in _SQL_SPECIAL_CHARACTERS:
        if special_char in ret_name:
            ret_name = ret_name.replace(special_char, "\\%s" % special_char)
    return ret_name


def connect(host, user, password, database = None):
    ret = None
    try:
        if database == None:
            ret = connector.connect(host = host, user = user, password = password)
        else:
            ret = connector.connect(host = host, user = user, password = password, database = database)
    except SQLError as e:
        ret = [False, e]
    return [True,ret]


class SqlTableColumn:

    def __init__(self, column_name: str, column_type: str, column_flags: Optional[str] = None):
        self.name = column_name
        self.col_type = column_type
        if column_flags is None:
            self.column_flags = ""
        else:
            self.column_flags = column_flags
