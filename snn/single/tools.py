import random as rd 
import string 
import matplotlib.pyplot as plt 
from math import *

class Callable_Float(float):
    def __new__(self, value):
        return float.__new__(self, value)
    def __call__(self, *args, **kwargs): return self

def create_equation(expected_args, eq):
    str_expected = ','.join('%s=None' % arg for arg in expected_args)
    return eval('lambda %s:%s' % (str_expected, eq)) if eq is not None else None

def create_bool_equation(expected_args, eq):
    str_expected = ','.join('%s=None' % arg for arg in expected_args)
    return eval('lambda %s: True if %s else False' % (str_expected, eq)) if eq is not None else None
 
def array_abs(arg):
    try : return abs(arg)
    except : 
        try :
            return [abs(item) for item in arg]
        except : 
            return None
