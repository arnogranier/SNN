import random as rd
from random import *
import math
from math import *
import numpy as np
from numpy import *


class Callable_Float(float):

    """Float that return its value when called"""

    def __new__(self, value):
        return float.__new__(self, value)

    def __call__(self, *args, **kwargs): return self


def create_equation(expected_args, eq):
    """Create equation with kwargs=expected args and return the value of eq"""
    str_expect = ','.join('%s=None' % arg for arg in expected_args)
    return eval('lambda %s:%s' % (str_expect, eq)) if eq is not None else None


def array_abs(arg):
    """Return the abs value of an array"""
    try:
        return abs(arg)
    except:
        try:
            return [abs(item) for item in arg]
        except:
            return None
