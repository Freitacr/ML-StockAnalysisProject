'''
Created on Dec 20, 2017

@author: Colton Freitas
'''

from .mysql_utils.mysql_utils import connect as SQLConnect
from general_utils.mysql_management.mysql_utils import mysql_utils
from mysql.connector.errors import InterfaceError


class MYSQLDataManipulator:

    def __init__(self, host, user, password, database=None):

        self.connection = SQLConnect(host, user, password, database)
        
    def insert_into_table(self, table, column_names, data, database = None):
        ''' Uploads the information in data into the table specified
        
        @param column_names: List containing the names of the slots to put each column of data into
        @param database: The database the table is stored in, if this is None, then the database currently
            focused by the connection is used. The database used is kept between calls to this method.
        @warning: Unfinished, and not fit for use as of yet
        TODO: Finish method
        
        '''
        
        cursor = self.connection.cursor()
        if not database == None:
            self.__switch_database(database, cursor)
        
        col_string = ",".join(column_names)
        
        insertion_sql = "INSERT INTO %s (%s) VALUES" % (table, col_string)
        insertion_sql += '(' + (','.join(['%s'] * len(column_names))) + ')'
        for to_insert in data:
            cursor.execute(insertion_sql, to_insert)
        
        
        self.__close_cursor(cursor)
        
    def __close_cursor(self, cursor):
        if not cursor.close():
            self.connection.close()
            raise ConnectionError("Cursor refused to close, cleaning up and exiting")
        
    def create_table(self, table_name, columns, database = None):
        ''' Creates a table in the database specified 
        
        @param columns: List of lists containing strings of the slot's name, followed by its type, and then any extra parameters needed
            for the slot creation. (I.E. 'primary key', 'auto_increment', etc)
        @param database: The database to create a new table in, if value is None, then uses the currently used database
        @warning: Unfinished and not fit for use
        TODO: turn columns into one string that can be used for the column declaration in table_creation_sql.
        
        '''
        
        #assuming parameter columns is in the form of [ ['id', 'int', 'primary key', 'auto_increment'], ['col1', 'text'] ... ]
        
        cursor = self.connection.cursor()
        
        if not database == None:
            self.__switch_database(database, cursor)
        
        column_declarations = []
        
        for column_declaration in columns:
            column_declarations.append(" ".join(column_declaration))
        
        columnString = ",".join(column_declarations)
        
        #TODO: turn columns into one string that can be used for the column declaration in table_creation_sql.
        
        table_creation_sql = "create table %s (%s)" % (table_name, columnString)
        
        
        cursor.execute(table_creation_sql)
        
        self.__close_cursor(cursor)

    def drop_table(self, table_name):
        sql = 'drop table %s' % table_name
        cursor = self.connection.cursor()
        cursor.execute(sql)
        self.__close_cursor(cursor)

    def show_tables_like(self, table_name):
        sql = 'show tables like "%s"' % mysql_utils.escape_table_name(table_name)
        cursor = self.connection.cursor()
        cursor.execute(sql)
        ret_iter = cursor.fetchall()
        self.__close_cursor(cursor)
        return ret_iter

    def __switch_database(self, database, cursor):
        cursor.execute("USE %s" % database)
        
    def select_from_table(self, table_name, column_list, database = None, conditional = None):
        column_string = ",".join(column_list)
        
        sql = "select %s from %s" % (column_string, table_name)
        
        if not conditional == None:
            sql += " %s" % conditional
        
        cursor = self.connection.cursor()
        
        if not database == None:
            self.__switch_database(database, cursor)
        
        cursor.execute(sql)
        ret_iter = cursor.fetchall()
        self.__close_cursor(cursor)
        return ret_iter

    def update(self, table_name: str, update: str, conditional: str):
        update_sql = f"update {table_name} set {update} where {conditional}"
        cursor = self.connection.cursor()
        cursor.execute(update_sql)
        self.__close_cursor(cursor)

    def explain(self, table_name):
        sql = "explain %s" % table_name
        cursor = self.connection.cursor()
        cursor.execute(sql)
        ret_iter = cursor.fetchall()
        self.__close_cursor(cursor)
        return ret_iter

    def execute_sql(self, sql):
        '''Method to execute a piece of SQL code directly, more for niche usage than normal use'''
        cursor = self.connection.cursor()
        
        cursor.execute(sql)
        ret_iter = None
        try:
            ret_iter = cursor.fetchall()
        except InterfaceError:
            pass
        self.__close_cursor(cursor)
        return ret_iter
    
    def commit(self):
        self.connection.commit()
    
    def rollback(self):
        self.connection.rollback()
        
    def close(self, commit=True):
        if commit:
            self.connection.commit()
        self.connection.close()