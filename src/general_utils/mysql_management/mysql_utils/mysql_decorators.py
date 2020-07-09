import functools
from typing import Callable, Optional, Union, Dict, List

from general_utils.mysql_management.mysql_data_manipulator import MYSQLDataManipulator
from general_utils.mysql_management.mysql_utils import mysql_utils
from general_utils.mysql_management.mysql_utils.mysql_utils import SqlTableColumn
from general_utils.mysql_management.mysql_utils import mysql_manipulator_function_factories
from general_utils.mysql_management.mysql_utils import mysql_table_abc


class TableLinkingError(BaseException):
    pass


class TableColumnMismatchError(TableLinkingError):
    pass


_MYSQL_CONVERSION_DICTIONARY = {
    "long": "mediumtext",
    "bool": "tinyint(1)",
    "boolean": "tinyint(1)"
}


def _is_field_type(program_field_type: str, described_field_type: str) -> bool:
    test_type_lower = program_field_type.lower()
    if test_type_lower in _MYSQL_CONVERSION_DICTIONARY:
        test_type_lower = _MYSQL_CONVERSION_DICTIONARY[test_type_lower]
    return described_field_type.startswith(test_type_lower)


def _do_flags_match(column_flags: str, descr_key: str, descr_extra: str) -> Union[bool, str]:
    if column_flags is None:
        return True
    lower_flags = column_flags.lower()
    if "primary key" in lower_flags:
        if not descr_key == "PRI":
            return "Table %s did not list column %s as a primary key, but the programmatic definition did."
    elif descr_key == "PRI":
        return "Table %s listed column %s as a primary key, but the programmatic definition did not."

    if "auto_increment" in lower_flags:
        if "auto_increment" not in descr_extra:
            return "Table %s did not list column %s as auto incrementing, but the programmatic definition did."
    elif "auto_increment" in descr_extra:
        return "Table %s listed column %s as auto incrementing, but the programmatic version did not."
    return True


def _do_tables_match(data_man: MYSQLDataManipulator, table_name: str, table_columns: Dict[str, SqlTableColumn]):
    for field, field_type, is_null, key_type, default, extra in data_man.explain(table_name):
        if field not in table_columns:
            raise TableColumnMismatchError("Table %s in database contained field %s "
                                           "but the programmatic definition did not." % (table_name, field))
        prog_col = table_columns[field]
        if not _is_field_type(prog_col.col_type, field_type):
            raise TableColumnMismatchError("Table %s's definition of column %s is of type %s, but the"
                                           "programmatic definition is of type %s" %
                                           (table_name, field, field_type, prog_col.col_type))
        res = _do_flags_match(prog_col.column_flags, key_type, extra)
        if isinstance(res, str):
            raise TableColumnMismatchError(res % table_name, field)


def _create_table(data_man: MYSQLDataManipulator, table_name: str, table_columns: Dict[str, SqlTableColumn]):
    column_definitions = []
    col_str_base = "%s %s %s"
    for column_name, column_def in table_columns.items():
        flags = "" if column_def.column_flags is None else column_def.column_flags
        column_definitions.append(
            (col_str_base % (column_def.name, column_def.col_type, flags), )
        )
    data_man.create_table(table_name, column_definitions)


def _bind_sql_functions(obj, table_name, column_defs: List[SqlTableColumn]):
    obj.drop_table = mysql_manipulator_function_factories.construct_drop_table_func(table_name)
    default_insertion_columns = [col_def for col_def in column_defs if "auto_increment" not in col_def.column_flags]
    obj.insert_into_table = mysql_manipulator_function_factories.construct_insert_into_table_func(
        table_name,
        [col.name for col in default_insertion_columns]
    )
    obj.select_from_table = mysql_manipulator_function_factories.construct_select_from_table_func(
        table_name,
        [col.name for col in column_defs]
    )
    obj.update = mysql_manipulator_function_factories.construct_update_func(table_name)


def mysql_table(fixed_table_name: Optional[str] = None):
    def mysql_table_decorator(init_function: Callable):
        """Designates an __init__ function that should be used to verify the existence and columns of a MySQL Table

        Args:
            init_function: class __init__ function.
            fixed_table_name: name of the SQL table the decorator should search for.
                If this is None, then the initializer must set self.db_tablename to the name of the table.
        """
        @functools.wraps(init_function)
        def table_linking(obj: mysql_table_abc.MySQLTable, *args, **kwargs):
            data_manipulator = mysql_manipulator_function_factories.AUTO_CLOSING_DATA_MANAGER
            # use instance variables of type SqlColumn to create table or verify table existence.
            table_columns = {}
            init_function(obj, *args, **kwargs)

            for attr_key in obj.__dict__:
                attr_val = obj.__dict__[attr_key]
                if isinstance(attr_val, mysql_utils.SqlTableColumn):
                    table_columns[attr_val.name] = attr_val

            if fixed_table_name is not None:
                obj.db_tablename = fixed_table_name
            elif not hasattr(obj, "db_tablename"):
                raise ValueError("Decorator did not have a table name passed in "
                                 "and initialization did not set db_tablename")
            table_name = obj.db_tablename
            if data_manipulator.show_tables_like(table_name):
                _do_tables_match(data_manipulator, table_name, table_columns)
            else:
                _create_table(data_manipulator, table_name, table_columns)
            _bind_sql_functions(obj, table_name, [col for col in table_columns.values()])
        return table_linking
    return mysql_table_decorator


