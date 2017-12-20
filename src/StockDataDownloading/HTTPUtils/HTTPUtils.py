'''
Created on Dec 19, 2017

@author: colton
'''
import urllib.request as ureq


def openURL(url):
    '''Opens the URL specified with basic error reporting'''
    hres = ureq.urlopen(url)
    if not str(hres.getcode())[0] == '2':
        return [False, hres.getcode()]
    else:
        return [True, hres]