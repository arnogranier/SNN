import random as rd 
import string 
import matplotlib.pyplot as plt 
from math import *

class Function:
    def __init__(self, arguments, foo):
        if arguments is None: self.arguments = ['t', ]
        else: 
            try :
                iter(arguments)
                self.arguments = arguments
            except: self.arguments = [arguments, ]
        if callable(foo) : self.foo = foo
        else: self.foo = eval('lambda %s : %s' % (','.join(arguments), foo)) 

    def __call__(self, *args) : return self.foo(*args)

def create_equation(variables, parameters, eq, bool_eq=False):
    var_str = ','.join([var.__name__ for var in sorted(variables, key=lambda x:x.__name__)])
    no_call_params_str = ','.join(['%s=%s' % (name, val) for (name, val) in parameters.items() if not callable(val)])
    call_params_str = ''
    callable_variables = {}
    for (name, val) in parameters.items() : 
        if isinstance(val, Function) : 
            name_var = ''.join(rd.choice(string.ascii_lowercase) for x in range(4))
            while name_var in [var.__name__ for var in variables]:
                name_var = ''.join(rd.choice(string.ascii_lowercase) for x in range(4))
            callable_variables[name_var] = val.foo
            call_params_str += '%s=callable_variables["%s"],' % (name, name_var)
            if eq is not None :eq = eq.replace(name, '%s(%s)' % (name, ','.join(val.arguments)))
    if bool_eq : 
        return eval('lambda t, %s,%s,%s: True if %s else False' % (var_str, no_call_params_str, call_params_str[:-1], eq))
    else:
        return eval('lambda t, %s,%s,%s:%s' % (var_str, no_call_params_str, call_params_str[:-1], eq))


def linspace(start, end, nb_points):
    pas = (1+end-start)/(nb_points)
    return [start+i*pas for i in range(round(nb_points))]

def meshgrid(x_data, y_data):
    x_nb_point, y_nb_point = (1+x_data[1]-x_data[0])/(x_data[2]), (1+y_data[1]-y_data[0])/(y_data[2])
    Y = [[y for _ in range(round(x_nb_point))] for y in linspace(y_data[0], y_data[1], y_nb_point)]
    X = [[x for x in linspace(x_data[0], x_data[1], x_nb_point)] for y in linspace(y_data[0], y_data[1], y_nb_point)]
    return X, Y

def array_abs(arg):
    try : return abs(arg)
    except : 
        try :
            return [abs(item) for item in arg]
        except : return None
