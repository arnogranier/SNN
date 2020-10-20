import matplotlib.pyplot as plt
from .tools import create_equation, array_abs, Callable_Float
from .numerical_methods import explicit_euler_step, rk4_step
import numpy as np


class Variable:
    """Represents a variable in the model.

    Parameters
    ----------
    name : str
    ddt : str
        derivation of the variable wrt to time
    init_value : float
        Initial value of membrane potential
    reset_value : float
        Reset value of membrane potential
    unit : str
        Unit of the variable

    """

    def __init__(self, name=None, ddt=None, init_value=-65,
                 reset_value=None, unit=None):
        if isinstance(name, str):
            self.__name__ = name
        else:
            alphabet = list('abcdefghijklmnopqrstuvwxyz')
            self.__name__ = ''.join(np.random.choice(alphabet)
                                    for x in range(3))
        self.unit = unit
        self.temp_ddt = ddt
        self.temp_reset_value = reset_value
        self.init_value = init_value

    def __str__(self):
        return self.__name__ + str(self.value)

    def __repr__(self):
        return str(self)

    def __call__(self, *args, **kwargs):
        return self.value


class Parameter:
    """Represents a parameter in the model.

    Parameters
    ----------
    name : str
    eq : str, float
        float or function of variables where the name of the variable in str
        form where the name of the variable can be stated in the str i.e.
        '0.07*exp((-V-65)/20)'
    expected_params : list
        List of expected variables in the parameter's equations
    unit : str

    Attributes
    ----------
    func : function
        Lambda expression created from the str form of equation

    """

    def __init__(self, name=None, eq=None, expected_params=[], unit=None):
        self.__name__ = name
        self.func = create_equation(expected_params, eq)
        self.unit = unit

    def __call__(self, *args, **kwargs):
        return self.value


class Model:
    """Short summary.

    Parameters
    ----------
    variables : list of Variable
        The list of variables in the model
    max_spike_value : float
        Maximum authorized value for the membrane potential. If higher the
        value is reseted to max_spike_value
    time_unit : str
    spike_when : str
        boolean equation describing the spiking event
    simul_method : str
        Name of the simulation method, either explicit_euler or rk4
    **parameters : dict
        Dict of parameters

    Attributes
    ----------
    expected_params : list of str
        List of the names of the expected parameters for variable and
        parameters update
    variables : dict of Variable
        Dict containing all the model's variables indexed by names
    variables_with_reset : list of Variable
        A list of variables with reset mechanisms
    parameters : dict of Parameters
        Dict containing all the model's parameters indexed by names
    _var_and_par : dict of Variable and Parameter
        Dict containing all variables AND paramteres indexed by names
    method : function
        Name of the simulation method (e.g explicit euler, rk4)
    step_method : type
        Simulation method step as defined in numerical_methods.py or custom
    """
    def __init__(self, *variables, max_spike_value=np.inf, time_unit='ms',
                 spike_when='False', simul_method='rk4', **parameters):

        # List of names of all the model variables and parameters + time
        self.expected_params = ['t'] + [var.__name__ for var in variables]  \
                               + list(parameters.keys())

        # Create equation for variable's ddts and reset_values
        for var in variables:
            var.ddt = create_equation(self.expected_params, var.temp_ddt)
            var.reset_value = create_equation(self.expected_params,
                                              var.temp_reset_value)

        # Create equation for model's representation of spikes
        self.spike_when = create_equation(self.expected_params, spike_when)
        self.max_spike_value = max_spike_value

        # Variables
        self.variables = {var.__name__: var for var in variables}
        self.variables_with_reset = [var for var in variables
                                     if var.reset_value is not None]

        # Parameters
        self.parameters = {name: Parameter(name=name, eq=str(val),
                                          expected_params=self.expected_params)
                           if not isinstance(val, Parameter) else val
                           for name, val in parameters.items()}

        # Store variables and parameters together in a dict
        self._var_and_par = {'t': None, **self.variables, **self.parameters}

        # Initialize variables, parameters and time
        self.reset()

        # Numerical simulation method
        self.method = simul_method
        if self.method == 'explicit_euler':
            self.step_method = explicit_euler_step
        elif self.method == 'rk4':
            self.step_method = rk4_step

        # Store unit of time
        self.time_unit = time_unit

    def reset(self):
        """Reset and initialize the model at its original state"""
        self._var_and_par['t'] = Callable_Float(0)

        for var in self.variables.values():
            var.value = var.init_value

        for_param_init = {'t': Callable_Float(0),
                          **{name: val()
                             for name, val in self.variables.items()}}
        for name, param in self.parameters.items():
            param.value = param.func(**for_param_init)

    @property
    def var_and_par(self):
        """Return a dict with values of variables and parameters + time"""
        return {name: val() for name, val in self._var_and_par.items()}

    def __setitem__(self, name, val):
        """Set variable or parameters with name name to be equal to val"""
        if name in self.parameters:
            self.parameters[name].func = create_equation(self.expected_params,
                                                         val)
        elif name in self.variables:
            self.variables[name].ddt = create_equation(self.expected_params,
                                                       val)
        self.reset()

    def simulation(self, T, dt, keep='all', start=dict()):
        """Simulate the model during T ms with time step dt ms, return
           - the history of variables and parmeters that are in keep as a dict
             {name(str) : values(numpy array)}
           - timming of spikes as a numpy array"""

        # If specified, initialize variable's values with values in start
        for name, var in self.variables.items():
            if name in start:
                var.value = start[name]

        # Initialize history and time_of_spikes
        history = {name: [val, ] for name, val in self.var_and_par.items()
                   if name in keep or keep == 'all'}
        time_of_spikes = []

        M = int(T / dt)
        state = self.var_and_par
        for p in range(M):

            # If we spiked last time step, update time_of_spikes and reset
            # variables that need to be
            if self.spike_when(**state):
                time_of_spikes.append(state['t'])
                for var in self.variables.values():
                    if var in self.variables_with_reset:
                        var.value = var.reset_value(**state)
                    else:
                        self.step_method(var, dt, state, self.max_spike_value)

            # Update variables
            else:
                for var in self.variables.values():
                    self.step_method(var, dt, state, self.max_spike_value)

            # Update parameters
            for name, param in self.parameters.items():
                param.value = param.func(**self.var_and_par)

            # Update time
            self._var_and_par['t'] = Callable_Float(p * dt)

            # Next state
            state = self.var_and_par

            # Update history
            for name, l in history.items():
                l.append(state[name])

        return ({name: np.array(vals) for name, vals in history.items()},
                np.array(time_of_spikes))

    def label(self, name, axis='y'):
        """Label an axe with the variables with name name, taing unit
           into account"""
        unit = self._var_and_par[name].unit
        if unit is not None:
            if axis == 'y':
                plt.ylabel('%s (%s)' % (name, unit))
            else:
                plt.xlabel('%s (%s)' % (name, unit))
        else:
            if axis == 'y':
                plt.ylabel('%s' % name)
            else:
                plt.xlabel('%s' % name)

    def plot(self, T=1000, dt=0.1, history=None, keep='all',
             subplotform=None, **kwargs):
        """Plot variables and/or parameters in keep against time, return
           a matplotlib figure"""

        x = np.linspace(0, T, T / dt + 1)

        # If history is None or there are missing var/params,
        # then simulate, else keep history
        if history is None or any(name not in history for name in keep):
            history, _ = self.simulation(T, dt, keep=keep)
        else:
            history = {var: vals for var, vals in history.items()
                       if var in keep}

        fig = plt.figure()
        for idx, (name, y) in enumerate(history.items()):
            if name != 't':
                # Subplot if needed
                if subplotform is not None:
                    plt.subplot(subplotform + str(idx + 1))
                    self.label(name, axis='y')

                # Plot data
                plt.plot(x, y, label=name, **kwargs)

                # Label x-axis
                plt.xlabel('time (%s)' % self.time_unit)

        # Labels/Legend for no-subplot graph
        if subplotform is None:
            if len(history) != 1:
                plt.legend()
            else:
                name = list(history.keys())[0]
                self.label(name, axis='y')

        plt.tight_layout()

        return fig

    def phase_portrait(self, xvarname, yvarname, T=1000, dt=0.1, history=None,
                       **kwargs):
        """Plot yvarname against xvarname"""

        # If history is None or there are missing var/params,
        # then simulate, else keep history
        if (history is None or
            (xvarname not in history or yvarname not in history)):
            history, _ = self.simulation(T, dt, keep=[xvarname, yvarname])

        fig = plt.figure()

        # Labels
        self.label(self.variables[xvarname].__name__, axis='x')
        self.label(self.variables[yvarname].__name__, axis='y')

        # Plot data
        plt.plot(history[xvarname], history[yvarname], **kwargs)

        return fig

    def phase_plane(self, xdata, ydata, others=dict(), nb_of_vector_by_axis=25,
                   rescale=False, no_dynamics=False, interactive=False, T=1000,
                   dt=0.2, quiver_args=dict(), contour_args=dict()):
        """Create a phase plan of two variables, with an automatic generation
           of the vector fields, and an interactive functionality
           xvardata and yvardata are feed as a 3-length tuple :
           (name, axis_min_Modelvalue, axis_max_value)"""

        # Set value of other variables and parameters if needed
        for name, val in others.items():
            self[name] = val

        # Number of vectors by axis
        x_nb_point = y_nb_point = nb_of_vector_by_axis
        x_name, y_name = xdata[0], ydata[0]

        # Get the derivate throught time of the 2 variables
        xvarddt = self.variables[x_name].ddt
        yvarddt = self.variables[y_name].ddt

        # Grid representation
        X, Y = np.meshgrid(np.linspace(xdata[1], xdata[2], x_nb_point),
                           np.linspace(ydata[1], ydata[2], y_nb_point))

        # Init dx and dy, that are vectors containing dx and dy for
        # each point needed
        dx = [[0 for _ in range(int(x_nb_point))]
              for _ in range(int(y_nb_point))]
        dy = [[0 for _ in range(int(x_nb_point))]
              for _ in range(int(y_nb_point))]

        # Fill dx and dy
        state = self.var_and_par
        for i in range(int(y_nb_point)):
            for j in range(int(x_nb_point)):

                # Get grid point
                x, y = X[i][j],  Y[i][j]

                # Update state with grid values
                state[x_name] = x
                state[y_name] = y

                # Fill dx and dy
                dx[i][j] = xvarddt(**state)
                dy[i][j] = yvarddt(**state)

        # If rescale, make y-variations equally important as x-variations,
        # by rescaling dy by fact
        if rescale:
            fact = sum([sum(array_abs(line)) for line in dx]) /   \
                   sum([sum(array_abs(line)) for line in dy])
            dy = [[fact*prev_dy for prev_dy in line] for line in dy]

        fig = plt.figure()

        # If we want to plot vector field, plot it
        if not no_dynamics:
            plt.quiver(X, Y, dx, dy, **quiver_args)

        # Plot nullclines by plotting the function wich verify dx = 0 or dy = 0
        cx = plt.contour(X, Y, dx, levels=[0], **contour_args)
        cy = plt.contour(X, Y, dy, levels=[0], **contour_args)

        # Label the nullclines
        plt.clabel(cx, fontsize=10, inline=1, inline_spacing=1,
                   fmt={0: '%s-nullcline' % x_name})
        plt.clabel(cy, fontsize=10, inline=1, inline_spacing=1,
                   fmt={0: '%s-nullcline' % y_name})

        # Label the axis
        self.label(xdata[0], axis='x')
        self.label(ydata[0], axis='y')

        # If the phase_plane is meant to be interactive, then ..
        if interactive:

            # Attach let click mouse button to the cascade method
            fig.canvas.mpl_connect('button_press_event',
                                   lambda evt: self.cascade(evt, line, point,
                                                            start_point, T, dt,
                                                            x_name, y_name))

            # Initialize line, mobile "current" point and starting point
            line, = plt.plot(list(), list(), '--')
            point, = plt.plot(list(), list(), 'o')
            start_point, = plt.plot(list(), list(), 'ro')

        return fig

    def cascade(self, event, line, point, start_point, T, dt, xvarname,
                yvarname):
        """When left mouse button pressed on interactive phase_plane,
           start plotting the dynamic (ie the evolution through time
           of the variables) """

        # Check that the click was really on the phase_plane
        if event.inaxes != line.axes:
            return

        # Get click coordinates
        x, y = event.xdata, event.ydata

        # Plot the starting point
        start_point.set_data(x, y)

        # Simulate the model starting with the right values for x and
        # y variables ie the coordinates of the click
        data, _ = self.simulation(T, dt, start={xvarname: x, yvarname: y})

        # Loop through the values of the variables during the simulation
        xs, ys = [x, ],  [y, ]
        fig = line.figure
        for (x, y) in [(x,y) for (x,y) in zip(data[xvarname], data[yvarname])]:

            # Update line
            xs.append(x)
            ys.append(y)
            line.set_data(xs, ys)

            # Update current point
            point.set_data(x, y)

            plt.pause(0.01)
            fig.canvas.draw()
