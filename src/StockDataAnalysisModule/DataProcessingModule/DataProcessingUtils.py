'''
Created on Dec 24, 2017

@author: Colton Freitas
'''


def percentChangeBetweenDays(day1, day2):
    if day1 == 0.0 and day2 == 0.0:
        return 0
    elif day1 == 0.0:
        return 100.0
    return ((day2 - day1) / day1) * 100.0
