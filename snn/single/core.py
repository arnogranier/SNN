import random as rd
import string
import math
import matplotlib.pyplot as plt
from .tools import create_equation, create_bool_equation, array_abs, Callable_Float
import numpy as np

class Variable:
    def __init__(self, name=None, ddt=None, init_value=-65,
                 reset_value=None, unit=None):
        if isinstance(name, str):
            self.__name__ = name
        else:
            self.__name__ = ''.join(rd.choice(string.ascii_lowercase)
                                    for x in range(3))
        self.unit = None
        self.temp_ddt = ddt
        self.temp_reset_value = reset_value
        self.value = init_value

    def __str__(self): return self.__name__ + str(self.value)
    def __repr__(self): return str(self)
    def __call__(self, *args, **kwargs): return self.value

class Parameter:
    def __init__(self, name=None, eq=None, expected_parameters=[]):
        self.__name__ = name
        self.func = create_equation(expected_parameters, eq)

    def __call__(self, *args, **kwargs): return self.value

class Model:
    def __init__(self, *variables, max_spike_value=math.inf,
                 spike_when='False', simul_method='rk4', **parameters):
        self.expected_parameters = ['t'] + [var.__name__ for var in variables] + list(parameters.keys())
        for var in variables:
            var.ddt = create_equation(self.expected_parameters, var.temp_ddt)
            var.reset_value = create_equation(self.expected_parameters, var.temp_reset_value)
        self.init_spike_when = spike_when
        self.spike_when = create_bool_equation(self.expected_parameters, spike_when)
        self.variables = {var.__name__ : var for var in variables}
        self.variables_with_reset = [var for var in variables
                                     if var.reset_value is not None]
        self.max_spike_value = max_spike_value
        self.parameters = {name:Parameter(name=name, eq=str(eq), expected_parameters=self.expected_parameters) 
                           for name, eq in parameters.items()}
        for_param_init = {'t':0, **{name:val() for name, val in self.variables.items()}}
        for name, param in self.parameters.items() : param.value = param.func(**for_param_init)
        self._var_and_par = {'t':Callable_Float(0), **self.variables, **self.parameters}
        self.method = simul_method

    @property
    def var_and_par(self):
        return {name : val() for name, val in self._var_and_par.items()}

    def __setitem__(self, name, val): 
        if name in self.parameters :
        	self.parameters[name].func = create_equation(self.expected_parameters, val)
        elif name in self.variables:
        	self.variables[name].ddt = create_equation(self.expected_parameters, val)

    def simulation(self, T, dt, keep='all', start=dict()):
        for name, var in self.variables.items():
            if name in start:
                var.value = start[name]

        history = {name:[val, ] for name, val in self.var_and_par.items() if name in keep or keep=='all'}
        if keep == 'all' or 't' in keep : history['t'] = [self.var_and_par['t'], ]
        
        M = int(T / dt)
        count_spike = 0
        for p in range(M):
            self._var_and_par['t'] = Callable_Float(p * dt)
            if self.spike_when(**self.var_and_par):
                count_spike += 1
                for var in self.variables_with_reset:
                    var.value = var.reset_value(**self.var_and_par)
            else:
                for name, var in self.variables.items():
                    if self.method == 'explicit_euler':
                        var.value = min(var() + dt * var.ddt(**self.var_and_par),
                                    self.max_spike_value)
                    elif self.method == 'rk4':
                        state, t, val = self.var_and_par.copy(), self.var_and_par['t'], var()
                        k1 = var.ddt(**state)
                        state['t'] = t + dt / 2 ; val = val + (dt / 2) * k1 ; state[name] = val 
                        k2 = var.ddt(**state)
                        val = val + (dt / 2) * k2 ; state[name] = val 
                        k3 = var.ddt(**state)
                        state['t'] = t + dt ; val = val + dt * k3 ; state[name] = val
                        k4 = var.ddt(**state)
                        var.value = min(var.value + (dt/6)*(k1+2*k2+2*k3+k4), self.max_spike_value)
            for name, param in self.parameters.items():
                param.value = param.func(**self.var_and_par)
            state = self.var_and_par
            for name, l in history.items(): l.append(state[name])

        return {name:np.array(vals) for name, vals in history.items()}, count_spike

    def plot(self, T, dt, history=None, keep='all',
             subplotform=None, **kwargs):
        x = np.linspace(0, T, T / dt + 1)
        if history is None:
            history, _ = self.simulation(T, dt, keep=keep)
        else:
            history = {var:vals for var, vals in history.items() if var in keep}
        fig = plt.figure()
        for idx, (name, y) in enumerate(history.items()):
        	if name != 't':
	            if subplotform is not None:
	                plt.subplot(subplotform + str(idx + 1))
	                unit = self.variables[name].unit
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
                plt.ylabel('%s (%s)' % (name, self.variables[name].unit))
        plt.tight_layout()
        return fig

    def plan_phase(self, xdata, ydata, other_variables=None,
                   rescale=False, no_dynamics=False, interactive=False, T=1000,
                   dt=1, quiver_args=dict(), contour_args=dict()):
        x_nb_point = (1 + xdata[2] - xdata[1]) / (xdata[3])
        y_nb_point = (1 + ydata[2] - ydata[1]) / (ydata[3])
        xvarddt = self.variables[xdata[0]].ddt
        yvarddt = self.variables[ydata[0]].ddt
        X, Y = np.meshgrid(np.linspace(xdata[1], xdata[2], x_nb_point),
                           np.linspace(ydata[1], ydata[2], y_nb_point))
        dx = [[0 for _ in range(int(x_nb_point))]
              for _ in range(int(y_nb_point))]
        dy = [[0 for _ in range(int(x_nb_point))]
              for _ in range(int(y_nb_point))]
        for i in range(int(y_nb_point)):
            for j in range(int(x_nb_point)):
                x, y = X[i][j],  Y[i][j]
                state = self.var_and_par
                state[xdata[0]] = x ; state[ydata[0]] = y
                dx[i][j] = xvarddt(**state)
                dy[i][j] = yvarddt(**state)
        if rescale:
            fact = sum([sum(array_abs(line)) for line in dx]) /   \
                   sum([sum(array_abs(line)) for line in dy])
            dy = [[fact*prev_dy for prev_dy in line] for line in dy]
        fig = plt.figure()
        if not no_dynamics:
            plt.quiver(X, Y, dx, dy, **quiver_args)
        cx = plt.contour(X, Y, dx, levels=[0], **contour_args)
        cy = plt.contour(X, Y, dy, levels=[0], **contour_args)
        plt.clabel(cx, fontsize=10, inline=1, inline_spacing=1, 
        	fmt={0:'%s-nullcline'%xdata[0]},
        	ticker=plt.LinearLocator())
        plt.clabel(cy, fontsize=10, inline=1, inline_spacing=1, 
        	fmt={0:'%s-nullcline'%ydata[0]},
        	ticker=plt.LinearLocator())
        plt.xlabel(xdata[0]) ; plt.ylabel(ydata[0])
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
