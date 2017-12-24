'''
Created on Dec 19, 2017

@author: Colton Freitas
'''
from ..HTTPUtils import HTTPUtils as utils




class CookieManager(object):
    '''Class for managing cookies for Yahoo's new download links
    
    This class just requires being instantiated for its job to be done. It will, however,
    raise an exception if there is a problem with obtaining either the cookie or the crumb.
    If no exception is raised, then the class methods will give the cookie and crumb respectively.
    '''


    def __init__(self):
        '''Constructor
        
        @raise ValueError: If connection to the URL fails, a ValueError will be raised from __getCookieandCrumb
        Constructor for the CookieManager class, also calls __getCookieandCrumb to finalize initialization.
        '''
        
        self.__cookie = None
        self.__crumb = None
        self.__crumbURL = 'https://finance.yahoo.com/quote/AAPL?p=AAPL'
        self.__getCookieandCrumb()
        
    def __getCookieandCrumb(self):
        '''Gets both the cookie and crumb from __crumbURL and stores them
        
        @raise ValueError: If the connection to __crumbURL fails (returns with an HTTP code that isn't in the format 2xx)
            a ValueError will be raised
        Obtains a cookie and crumb from Yahoo Finance, storing them inside respective class fields for retrieval
        '''
        
        stat = utils.openURL(self.__crumbURL)
        if not stat[0]:
            raise ValueError("Connection to {0} failed with code {1}".format(self.__crumbURL, stat[1]))
        httpresponse = stat[1]
        #Grab the cookie from the http header
        cookiestr = httpresponse.getheader('set-cookie')
        cookiestr = cookiestr.split('; expires')[0]
        self.__cookie = cookiestr
        
        #Grab the crumb from the webpage
        webpageLines = []
        for bline in  httpresponse:
            #unicode-escape is used as there are times when the unicode character
            #\u004 exists within the string. Unicode-escape handles it fine.
            linestr = bline.decode('unicode-escape')
            webpageLines.append(linestr)
        for line in webpageLines:
            index = line.find("Crumb")
            if not index == -1:
                endIndex = line.find("}", index)
                self.__crumb = line[index : endIndex + 1]
        self.__crumb = self.__crumb.split(":")[-1][:-2]
        
    def getCookie(self):
        '''Returns the cookie obtained during class initialization'''
        return self.__cookie
    def getCrumb(self):
        '''Returns the crumb obtained during class initialization'''
        return self.__crumb
    