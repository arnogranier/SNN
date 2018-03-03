import random as rd
import string
import math
import matplotlib.pyplot as plt
from .tools import create_equation, linspace, meshgrid, array_abs, Function
import numpy as np

class Variable:
    def __init__(self, name=None, ddt='0', init_value=-65,
                 reset_value=None, unit=None, forbidden_values=None):
        if isinstance(name, str):
            self.__name__ = name
        else:
            self.__name__ = ''.join(rd.choice(string.ascii_lowercase)
                                    for x in range(3))
        if isinstance(unit, str):
            self.unit = unit
        else:
            self.unit = None
        if forbidden_values is not None:
        	if callable(forbidden_values):
        		self.forbidden_values = forbidden_values
        	else:
        		self.forbidden_values = [forbidden_values, ]
        else:
        	self.forbidden_values = []
        self.ddt = lambda *args: 0
        self.temp_ddt = ddt
        self.temp_reset_value = reset_value
        self.value = init_value

    def __str__(self): return self.__name__ + str(self.value)

    def __repr__(self): return str(self)

    @property
    def value(self): return self._value if self._value not in self.forbidden_values else self._value + 0.01

    @value.setter
    def value(self, val):
        self._value = val 
    @property
    def unit(self): return self._unit

    @unit.setter
    def unit(self, val):
        if isinstance(val, str) or val is None:
            self._unit = val

    @property
    def ddt(self): return self._ddt

    @ddt.setter
    def ddt(self, foo):
        if callable(foo):
            self._ddt = foo

    @property
    def reset_value(self):
        return self._reset_value

    @reset_value.setter
    def reset_value(self, foo):
        if callable(foo) or foo is None:
            self._reset_value = foo


class Model:
    def __init__(self, *variables, max_spike_value=math.inf,
                 spike_when='False', simul_method='rk4', **parameters):
        self.init_spike_when = spike_when
        self.spike_when = create_equation(variables, parameters,
                                          spike_when, bool_eq=True)
        self.variables = sorted(variables, key=lambda x: x.__name__)
        self.variables_with_reset = [var for var in self.variables
                                     if var.temp_reset_value is not None]
        self.max_spike_value = max_spike_value
        self.parameters = parameters
        for var in variables:
            var.reset_value = create_equation(variables, parameters,
                                              var.temp_reset_value)
            var.ddt = create_equation(variables, parameters, var.temp_ddt)
        self.method = simul_method

    def __setitem__(self, name, val): 
        self.parameters[name] = val
        self.__init__(*self.variables, max_spike_value=self.max_spike_value, 
                      spike_when=self.init_spike_when, simul_method=self.method,
                      **self.parameters)

    def simulation(self, T, dt, keep='all', start=dict()):
        for var in self.variables:
            if var.__name__ in start:
                var.value = start[var.__name__]

        start_var_and_params = {var.__name__: var.value
                                for var in self.variables}
        for (name, value) in self.parameters.items():
            if callable(value):
                args = [0, ] if 't' in value.arguments else list() 
                for var in self.variables:
                    if var.__name__ in value.arguments: args.append(var.value)
                start_var_and_params[name] = value(*args)
            else:
                start_var_and_params[name] = value
        if keep == 'all':
            keep_parameters_name = [name for name in self.parameters]
        else:
            try:
                iter(keep)
            except:
                keep = [keep]
            keep_parameters_name = [name for name in keep
                                    if name in self.parameters]
        history = {name: [value, ]
                    for (name, value) in start_var_and_params.items()
                    if name in keep or keep == 'all'}
        M = int(T / dt)
        count_spike = 0
        for p in range(M):
            t = p * dt
            var_values = [var.value for var in self.variables]
            if self.spike_when(t, *var_values):
                count_spike += 1
                for var in self.variables_with_reset:
                    var.value = var.reset_value(t, *var_values)
                    if var.__name__ in history:
                        history[var.__name__].append(var.value)
            else:
                for var in self.variables:
                    if self.method == 'explicit_euler':
                        var.value = min(var.value + dt * var.ddt(t, *var_values),
                                    self.max_spike_value)
                    # mettre ddt sous la forme lambda :
                    elif self.method == 'rk4':
                        k1 = var.ddt(t, *var_values)
                        k2 = var.ddt(t+dt/2, *[v.value if v.__name__ != var.__name__ else v.value+(dt/2)*k1 for v in self.variables])
                        k3 = var.ddt(t+dt/2, *[v.value if v.__name__ != var.__name__ else v.value+(dt/2)*k2 for v in self.variables])
                        k4 = var.ddt(t+dt, *[v.value if v.__name__ != var.__name__ else v.value+dt*k3 for v in self.variables])
                        var.value = min(var.value + (dt/6)*(k1+2*k2+2*k3+k4), self.max_spike_value)
                    if var.__name__ in history:
                        history[var.__name__].append(var.value)
            for name in keep_parameters_name:
                val = self.parameters[name]
                if callable(val):
                    args = [t, ] if 't' in val.arguments else list() 
                    for var in self.variables:
                        if var.__name__ in val.arguments: args.append(var.value)
                    history[name].append(val(*args))
                else:
                    history[name].append(val)
        return {name:np.array(vals) for name, vals in history.items()}, count_spike

    def get_unit_from_name(self, name):
        for var in self.variables:
            if var.__name__ == name:
                return var.unit

    def get_ddt_from_name(self, name):
        for var in self.variables:
            if var.__name__ == name:
                return var.ddt

    def plot(self, T, dt, history=None, keep='all',
             subplotform=None, **kwargs):
        x = linspace(0, T, T / dt + 1)
        if history is None:
            history, _ = self.simulation(T, dt, keep=keep)
        else:
            history = {var:vals for var, vals in history.items() if var in keep}
        fig = plt.figure()
        for idx, (name, y) in enumerate(history.items()):
            if subplotform is not None:
                plt.subplot(subplotform + str(idx + 1))
                unit = self.get_unit_from_name(name)
                if unit is not None:
                    plt.ylabel('%s (%s)' % (name, unit))
                else:
                    plt.ylabel('%s' % name)
            plt.plot(x, y, label=name, **kwargs)
            plt.xlabel('time')
        if subplotform is None:
            if len(history) != 1:
                plt.legend()
            else:
                name = list(history.keys())[0]
                plt.ylabel('%s (%s)' % (name, self.get_unit_from_name(name)))
        plt.tight_layout()
        return fig

    def plan_phase(self, xdata, ydata, other_variables=None,
                   rescale=False, no_dynamics=False, interactive=False, T=1000,
                   dt=1, quiver_args=dict(), contour_args=dict()):
        x_nb_point = (1 + xdata[2] - xdata[1]) / (xdata[3])
        y_nb_point = (1 + ydata[2] - ydata[1]) / (ydata[3])
        xvarddt = self.get_ddt_from_name(xdata[0])
        yvarddt = self.get_ddt_from_name(ydata[0])
        X, Y = meshgrid(xdata[1:], ydata[1:])
        dx = [[0 for _ in range(int(x_nb_point))]
              for _ in range(int(y_nb_point))]
        dy = [[0 for _ in range(int(x_nb_point))]
              for _ in range(int(y_nb_point))]
        for i in range(int(y_nb_point)):
            for j in range(int(x_nb_point)):
                x, y = X[i][j],  Y[i][j]
                var = [x if name == xdata[0] else y if name == ydata[0] else 0
                       for name in [var.__name__
                                    for var in sorted(self.variables,
                                                    key=lambda x: x.__name__)]]
                dx[i][j] = xvarddt(0, *var)
                dy[i][j] = yvarddt(0, *var)
        if rescale:
            fact = sum([sum(array_abs(line)) for line in dx]) /   \
                   sum([sum(array_abs(line)) for line in dy])
            dy = [[fact*prev_dy for prev_dy in line] for line in dy]
        fig = plt.figure()
        if not no_dynamics:
            plt.quiver(X, Y, dx, dy, **quiver_args)
        plt.contour(X, Y, dx, levels=[0], **contour_args)
        plt.contour(X, Y, dy, levels=[0], **contour_args)
        if interactive:
            fig.canvas.mpl_connect('button_press_event',
                                   lambda evt: self.cascade(evt, line, point,
                                        start_point, T, dt, xdata[0], ydata[0],
                                        other_variables=other_variables))
            line, = plt.plot(list(), list(), '--')
            point, = plt.plot(list(), list(), 'o')
            start_point, = plt.plot(list(), list(), 'ro')
        return fig

    def cascade(self, event, line, point, start_point, T, dt, xvarname,
                yvarname, other_variables=None):
        if event.inaxes != line.axes:
            return
        x, y = event.xdata, event.ydata
        start_point.set_data(x, y)
        xs, ys = [x, ],  [y, ]
        data, _ = self.simulation(T, dt, start={xvarname: x, yvarname: y})
        for (x, y) in [(x,y) for (x,y) in zip(data[xvarname], data[yvarname])]:
            xs.append(x)
            ys.append(y)
            line.set_data(xs, ys)
            point.set_data(x, y)
            plt.pause(0.01)
            line.figure.canvas.draw()
