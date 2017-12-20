'''
Created on Dec 19, 2017

@author: Colton Freitas
'''
import urllib.request as ureq
from GeneralUtils.EPrint import eprint
from urllib.error import HTTPError


def openURL(url, cookie = None):
    '''Opens the URL specified with basic error reporting'''
    opener = ureq.build_opener()
    if not cookie == None:
        print(cookie)
        opener.addheaders.append( ('Cookie', cookie))
    hres = None
    try:
        hres = opener.open(url)
    except HTTPError as e:
        eprint(str(e))
        return [False, str(e)]
    if not str(hres.getcode())[0] == '2':
        return [False, hres.getcode()]
    else:
        return [True, hres]