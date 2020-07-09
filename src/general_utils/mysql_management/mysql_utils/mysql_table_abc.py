import abc
from typing import List, Optional, Iterable, Any

from general_utils.mysql_management import mysql_data_manipulator


class MySQLTable(abc.ABC):

    @abc.abstractmethod
    def drop_table(self):
        pass

    @abc.abstractmethod
    def select_from_table(self,
                          column_list: List[str],
                          database: Optional[str] = None,
                          conditional: Optional[str] = None):
        pass

    @abc.abstractmethod
    def insert_into_table(self,
                          data: List[Iterable[Any]],
                          column_names: Optional[List[str]] = None,
                          database: Optional[str] = None
                          ):
        pass

    @abc.abstractmethod
    def update(self,
               update: str,
               conditional: str):
        pass

