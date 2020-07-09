import abc
from typing import List, Optional, Iterable, Any

from general_utils.mysql_management import mysql_data_manipulator


class MySQLTable(abc.ABC):

    def __init__(self, data_manager: mysql_data_manipulator.MYSQLDataManipulator):
        # paramaterized empty constructor is to strongly hint at the requirement of a table needing a
        # data manipulator in its argument list
        pass

    @abc.abstractmethod
    def drop_table(self, data_manager: mysql_data_manipulator.MYSQLDataManipulator):
        pass

    @abc.abstractmethod
    def select_from_table(self,
                          data_manager: mysql_data_manipulator.MYSQLDataManipulator,
                          column_list: List[str],
                          database: Optional[str] = None,
                          conditional: Optional[str] = None):
        pass

    @abc.abstractmethod
    def insert_into_table(self,
                          data_manager: mysql_data_manipulator.MYSQLDataManipulator,
                          data: List[Iterable[Any]],
                          column_names: Optional[List[str]] = None,
                          database: Optional[str] = None
                          ):
        pass

    @abc.abstractmethod
    def update(self,
               data_manager: mysql_data_manipulator.MYSQLDataManipulator,
               update: str,
               conditional: str):
        pass

