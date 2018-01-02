'''
Created on Jan 2, 2018

@author: Colton Freitas
'''
import sys


class ToleranceString:
    
    def __init__(self, direction, amounts, zero_state = "both"):
        self.__tol_str = self.__createToleranceString(direction, amounts)
        zero_states = ["positive", "negative", "both"]
        if not zero_state.lower() in zero_states:
            raise ValueError("Invalid selection of zero-state, must be 'positive', 'negative', or 'both'")
        self.__zero_state = zero_states.index(zero_state.lower())
    
    def __setupToleranceString(self, direction):
        valid_directions = ["less", "more", "extremes", "zero"]
        if direction not in valid_directions:
            print("%s not in valid directions, valid directions are: %s" % (direction, valid_directions), file=sys.stderr)
            return ""
        
        if direction == valid_directions[0]:
            return "-%.2f:-%.2f"
        elif direction == valid_directions[1]:
            return "+%.2f:+%.2f"
        elif direction == valid_directions[2]:
            return "-%.2f:+%.2f"
        elif direction == valid_directions[3]:
            return "+%.2f:-%.2f"
        else:
            raise ValueError("Invalid direction, this should have been caught.")
    
    def __createToleranceString(self, direction, amounts):
        tol_str = self.__setupToleranceString(direction)
        if tol_str == "":
            raise ValueError("Invalid Direction %s" % (direction))
        return tol_str % tuple(amounts)

    def test(self, expected, actual):
        if expected == 0:
            if self.__zero_state == 0:
                tol_amount = float(self.__tol_str.split(":")[0])
            elif self.__zero_state == 1:
                tol_amount = float(self.__tol_str.split(":")[1])
            elif self.__zero_state == 2:
                tol_amount = float(self.__tol_str.split(":")[0])
                neg_flag = (actual >= expected and actual <= expected + tol_amount) or (actual >= expected + tol_amount and actual <= expected)
                tol_amount = float(self.__tol_str.split(":")[1])
                pos_flag = (actual >= expected and actual <= expected + tol_amount) or (actual >= expected + tol_amount and actual <= expected)
                return pos_flag or neg_flag
            #DoZeroChecking
        elif expected < 0:
            tol_amount = float(self.__tol_str.split(":")[0])
            #One of these conditions will be impossible, this is fine. The other one will be the one to test against.
        elif expected > 0:
            tol_amount = float(self.__tol_str.split(":")[1])
        return (actual >= expected and actual <= expected + tol_amount) or (actual >= expected + tol_amount and actual <= expected)
