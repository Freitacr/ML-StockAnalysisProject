from typing import List, Iterable, Any, Optional

from general_utils.mysql_management import mysql_data_manipulator
from general_utils.mysql_management.mysql_utils import mysql_table_abc
from general_utils.mysql_management.mysql_utils import mysql_decorators
from general_utils.mysql_management.mysql_utils import mysql_utils


HISTORICAL_DATE_COLUMN_NAME = "hist_date"
HIGH_PRICE_COLUMN_NAME = "high_price"
LOW_PRICE_COLUMN_NAME = "low_price"
OPEN_PRICE_COLUMN_NAME = "opening_price"
CLOSING_PRICE_COLUMN_NAME = "close_price"
ADJUSTED_CLOSING_PRICE_COLUMN_NAME = "adj_close"
VOLUME_COLUMN_NAME = "volume_data"


class StockDataTable(mysql_table_abc.MySQLTable):

    @mysql_decorators.mysql_table(fixed_table_name=None)
    def __init__(self, table_name: str):
        super().__init__()
        if table_name is None:
            raise ValueError("table_name must not be None")
        self.db_tablename = table_name
        self.id = mysql_utils.SqlTableColumn("id", "int", "primary key auto_increment")
        self.hist_date = mysql_utils.SqlTableColumn(HISTORICAL_DATE_COLUMN_NAME, "Date")
        self.high = mysql_utils.SqlTableColumn(HIGH_PRICE_COLUMN_NAME, "float")
        self.low = mysql_utils.SqlTableColumn(LOW_PRICE_COLUMN_NAME, "float")
        self.opening = mysql_utils.SqlTableColumn(OPEN_PRICE_COLUMN_NAME, "float")
        self.closing = mysql_utils.SqlTableColumn(CLOSING_PRICE_COLUMN_NAME, "float")
        self.adj_close = mysql_utils.SqlTableColumn(ADJUSTED_CLOSING_PRICE_COLUMN_NAME, "float")
        self.vol = mysql_utils.SqlTableColumn(VOLUME_COLUMN_NAME, "long")

    def update(self, update: str, conditional: str):
        pass

    def drop_table(self):
        pass

    def select_from_table(self,
                          column_list: List[str],
                          database: Optional[str] = None,
                          conditional: Optional[str] = None):
        pass

    def insert_into_table(self,
                          data: List[Iterable[Any]],
                          column_names: Optional[List[str]] = None,
                          database: Optional[str] = None):
        pass
