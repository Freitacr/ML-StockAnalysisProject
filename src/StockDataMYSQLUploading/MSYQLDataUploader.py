'''
Created on Dec 20, 2017

@author: Colton Freitas
'''

from StockDataMYSQLUploading.MYSQLUtils.MYSQLUtils import connect as SQLConnect




class MYSQLDataUploader:
    '''
    
    #TODO: Change error handling in class to be more customized and informative
    '''


    def __init__(self, host, user, password, database=None):
        '''Constructor 
        
        
        
        '''
        connectionStatus = SQLConnect(host, user, password, database)
        if not connectionStatus[0]:
            raise ConnectionError(connectionStatus[1]) 
        self.connection = connectionStatus[1]
        
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
            self.__switch_database(database)
        
        
        
        
        if not cursor.close():
            self.connection.close()
            raise ConnectionError("Cursor refused to close, cleaning up and exiting")
        
    def create_table(self, table_name, columns, database = None):
        ''' Creates a table in the database specified 
        
        @param columns: List of tuples consisting of the slot's name, followed by its type, and then any extra parameters needed
            for the slot creation. (I.E. 'primary key', 'auto_increment', etc)
        @param database: The database to create a new table in, if value is None, then uses the currently used database
        @warning: Unfinished and not fit for use
        TODO: turn columns into one string that can be used for the column declaration in table_creation_sql.
        
        '''
        
        cursor = self.connection.cursor()
        
        if not database == None:
            self.__switch_database(database)
        
        
        columnString = ""
        
        #TODO: turn columns into one string that can be used for the column declaration in table_creation_sql.
        
        table_creation_sql = "create table %s, (%s)" % (table_name, columnString)
        
        
        
        
        
        
        
        
        
        
        
        
        
        

    def __switch_database(self, database):
        cursor = self.connection.cursor()
        cursor.execute("USE %s" % database)
        cursor.close()