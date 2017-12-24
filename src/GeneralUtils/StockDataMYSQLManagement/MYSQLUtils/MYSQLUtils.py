'''
Created on Dec 20, 2017

@author: Colton Freitas
'''

import mysql.connector as connector
from mysql.connector.errors import Error as SQLError
from GeneralUtils.EPrint import eprint

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