'''
Created on Dec 19, 2017

@author: Colton Freitas
'''
import urllib.request as ureq
from urllib.error import HTTPError


def openURL(url, cookie = None):
    '''Opens the URL specified with basic error reporting'''
    opener = ureq.build_opener()
    if not cookie == None:
        opener.addheaders.append( ('Cookie', cookie))
    hres = None
    try:
        hres = opener.open(url)
    except HTTPError as e:
        return [False, (e)]
    if not str(hres.getcode())[0] == '2':
        return [False, hres.getcode()]
    else:
        return [True, hres]