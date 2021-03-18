from typing import List, Iterable, Any, Optional

from general_utils.mysql_management import mysql_data_manipulator
from general_utils.mysql_management.mysql_utils import mysql_table_abc
from general_utils.mysql_management.mysql_utils import mysql_decorators
from general_utils.mysql_management.mysql_utils import mysql_utils


TICKER_COLUMN_NAME = "ticker"
YAHOO_COLUMN_NAME = "yahoo"
GOOGLE_COLUMN_NAME = "google"


class StockListTable(mysql_table_abc.MySQLTable):

    @mysql_decorators.mysql_table(fixed_table_name="stock_list")
    def __init__(self):
        super().__init__()
        self.id = mysql_utils.SqlTableColumn("id", "int", "primary key auto_increment")
        self.ticker = mysql_utils.SqlTableColumn(TICKER_COLUMN_NAME, "text")
        self.yahoo = mysql_utils.SqlTableColumn(YAHOO_COLUMN_NAME, "bool")
        self.google = mysql_utils.SqlTableColumn(GOOGLE_COLUMN_NAME, "bool")

    def update(self, update: str, conditional: str):
        pass

    def drop_table(self):
        pass

    def select_from_table(self, column_list: List[str],
                          database: Optional[str] = None, conditional: Optional[str] = None):
        pass

    def insert_into_table(self, data: List[Iterable[Any]],
                          column_names: Optional[List[str]] = None, database: Optional[str] = None):
        pass
