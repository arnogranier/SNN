import random as rd 
import string 
import matplotlib.pyplot as plt 

def create_equation(variables, parameters, eq, bool_eq=False):
    var_str = ','.join([var.__name__ for var in sorted(variables, key=lambda x:x.__name__)])
    no_call_params_str = ','.join(['%s=%s' % (name, val) for (name, val) in parameters.items() if not callable(val)])
    call_params_str = ''
    for (name, val) in parameters.items() : 
        callable_variables = {}
        if callable(val) : 
            name_var = ''.join(rd.choice(string.ascii_lowercase) for x in range(4))
            while name_var in [var.__name__ for var in variables]:
                name_var = ''.join(rd.choice(string.ascii_lowercase) for x in range(4))
            callable_variables[name_var] = val
            call_params_str += '%s=callable_variables["%s"],' % (name, name_var)
            eq = eq.replace(name, '%s(t)' % name)
    if bool_eq : 
        return eval('lambda t, %s,%s,%s: True if %s else False' % (var_str, no_call_params_str, call_params_str[:-1], eq))
    else:
        return eval('lambda t, %s,%s,%s:%s' % (var_str, no_call_params_str, call_params_str[:-1], eq))


def linspace(start, end, nb_points):
    pas = (1+end-start)/(nb_points)
    return [start+i*pas for i in range(round(nb_points))]

def meshgrid(x_data, y_data):
    x_nb_point, y_nb_point = (1+x_data[1]-x_data[0])/(x_data[2]), (1+y_data[1]-y_data[0])/(y_data[2])
    Y = [[y for _ in range(round(y_nb_point))] for y in linspace(y_data[0], y_data[1], y_nb_point)]
    X = [[x for x in linspace(x_data[0], x_data[1], x_nb_point)] for _ in range(round(x_nb_point))]
    return X, Y

def array_abs(arg):
    try : return abs(arg)
    except : 
        try :
            return [abs(item) for item in arg]
        except : return None
