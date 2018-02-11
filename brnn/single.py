import random as rd 
import string
import math
import matplotlib.pyplot as plt
from .tools import create_equation, linspace, meshgrid, array_abs


class Variable:
    def __init__(self, name=None, ddt='0', init_value=-65, reset_value=None, unit=None):
        if isinstance(name, str) : self.__name__ = name
        else : self.__name__ =  ''.join(rd.choice(string.ascii_lowercase) for x in range(3))
        if isinstance(unit, str) : self.unit = unit
        else : self.unit = None
        self.ddt = lambda *args : 0
        self.temp_ddt = ddt
        self.temp_reset_value = reset_value
        self.value = init_value
    def __str__(self):return self.__name__ + str(self.value)
    def __repr__(self):return str(self)
    @property
    def value(self): return self._value
    @value.setter
    def value(self, val):
        self._value = val
    @property 
    def unit(self): return self._unit
    @unit.setter
    def unit(self, val) : 
        if isinstance(val, str) or val is None : self._unit = val
    @property
    def ddt(self) : return self._ddt
    @ddt.setter
    def ddt(self, foo) : 
        if callable(foo) : self._ddt = foo
    @property
    def reset_value(self) : return self._reset_value
    @reset_value.setter
    def reset_value(self, foo) : 
        if callable(foo) or foo is None : self._reset_value = foo


class Model:
    def __init__(self, *variables,max_spike_value=math.inf, spike_when='False', **parameters):
        self.spike_when = create_equation(variables, parameters, spike_when, bool_eq=True)
        self.variables = sorted(variables, key=lambda x:x.__name__)
        self.variables_with_reset = [var for var in self.variables if var.temp_reset_value is not None]
        self.max_spike_value = max_spike_value
        self.parameters = parameters
        for var in variables : 
            var.ddt = create_equation(variables, parameters, var.temp_ddt)
            var.reset_value = create_equation(variables, parameters, var.temp_reset_value)
    
    def simulation(self, T, dt, keep='all', start={}):
        for var in self.variables : 
            if var.__name__ in start :
                var.value = start[var.__name__] 
        if keep == 'all': 
            history = {var.__name__:[var.value, ] for var in self.variables}
            keep_parameters_name = {}
        else:
            try: iter(keep)
            except: keep = [keep] 
            start_var_and_params = {var.__name__ : var.value for var in self.variables}
            for (name, value) in self.parameters.items() : 
                if callable(value) : start_var_and_params[name] = value(0)
                else : start_var_and_params[name] = value
            keep_parameters_name = [name for name in keep if name in self.parameters]
            history = {name : [value, ] for (name, value) in start_var_and_params.items() if name in keep}
            
        M = int(T / dt)
        count_spike = 0
        for p in range(M):
            t = p*dt
            var_values = [var.value for var in self.variables]
            if self.spike_when(t, *var_values):
                count_spike += 1
                for var in self.variables_with_reset: 
                    var.value = var.reset_value(t, *var_values)
                    if var.__name__ in history : history[var.__name__].append(var.value)
            else:
                for var in self.variables:
                    var.value = min(var.value + dt * var.ddt(t, *var_values), self.max_spike_value)
                    if var.__name__ in history : history[var.__name__].append(var.value)
            for name in keep_parameters_name : 
                val = self.parameters[name]
                if callable(val) : 
                    history[name].append(val(t))
                else : 
                    history[name].append(val)
        return history, count_spike

    def get_unit_from_name(self, name):
        for var in self.variables : 
            if var.__name__ == name : return var.unit

    def get_ddt_from_name(self, name):
        for var in self.variables : 
            if var.__name__ == name : return var.ddt       

    def plot(self, T, dt, history=None, keep='all', 
                    subplotform=None, **kwargs):
        x = linspace(0, T, T / dt+1)
        if history is None : history,_ = self.simulation(T, dt, keep=keep)
        fig = plt.figure()
        for idx, (name, y) in enumerate(history.items()):
            if subplotform is not None :
                plt.subplot(subplotform + str(idx+1))
                unit = self.get_unit_from_name(name)
                if unit is not None : plt.ylabel('%s (%s)'% (name, unit))
                else : plt.ylabel('%s'% name)
            plt.plot(x, y, label=name, **kwargs)
            plt.xlabel('time')
        if subplotform is None :
            if len(history) != 1 : plt.legend()
            else : 
                name = list(history.keys())[0]
                plt.ylabel('%s (%s)' % (name, self.get_unit_from_name(name)))
        plt.tight_layout()
        return fig

    def plan_phase(self, xvardata, yvardata, other_variables=None, rescale=False, no_dynamics=False, 
                    interactive=False, T=1000, dt=1):
        x_nb_point, y_nb_point = (1+xvardata[2]-xvardata[1])/(xvardata[3]), (1+yvardata[2]-yvardata[1])/(yvardata[3])
        xvarddt, yvarddt = self.get_ddt_from_name(xvardata[0]), self.get_ddt_from_name(yvardata[0])
        X, Y = meshgrid(xvardata[1:], yvardata[1:])
        dx = [[0 for _ in range(round(y_nb_point))] for _ in range(round(x_nb_point))] 
        dy = [[0 for _ in range(round(y_nb_point))] for _ in range(round(x_nb_point))]
        for i in range(round(x_nb_point)):
            for j in range(round(y_nb_point)):
                x = X[i][j] ; y = Y[i][j]
                var = [x if name==xvardata[0] else y if name==yvardata[0] else 0
                       for name in [var.__name__ for var in sorted(self.variables, key=lambda x:x.__name__)]]
                dx[i][j] = xvarddt(0, *var)
                dy[i][j] = yvarddt(0, *var)
        if rescale :
            fact = sum([sum(array_abs(line)) for line in dx]) / sum([sum(array_abs(line)) for line in dy])
            dy = [[fact*prev_dy for prev_dy in line] for line in dy]
        fig = plt.figure()
        if not no_dynamics : plt.quiver(X, Y, dx, dy)
        plt.contour(X, Y, dx, levels=[0], linewidths=3, colors='black')
        plt.contour(X, Y, dy, levels=[0], linewidths=3, colors='black')
        if interactive:
            fig.canvas.mpl_connect('button_press_event', lambda evt:self.cascade(evt, line, point, start_point,
                                    T, dt, xvardata[0], yvardata[0], other_variables=other_variables))
            line, = plt.plot([], [], '--')
            point, = plt.plot([], [], 'o')
            start_point, = plt.plot([], [], 'ro')
        return fig

    def cascade(self, event, line, point, start_point, T, dt, xvarname, yvarname, other_variables=None):
        if event.inaxes != line.axes: return
        x = event.xdata ; y = event.ydata
        start_point.set_data(x, y)
        xs = [x, ] ; ys = [y, ]
        data, _ = self.simulation(T, dt, start={xvarname:x, yvarname:y})
        for (x, y) in [(x,y) for (x,y) in zip(data[xvarname], data[yvarname])]:
            xs.append(x) ; ys.append(y)
            line.set_data(xs, ys)
            point.set_data(x, y)
            plt.pause(0.01)
            line.figure.canvas.draw()
