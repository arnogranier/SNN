# All the functions here need to be of the form
# def foo(var, dt, state, max_spike_value):
#      update var value at some point (var.value = something)


def explicit_euler_step(var, dt, state, max_spike_value):
    """Explicit euler method step"""
    var.value = min(var() + dt * var.ddt(**state), max_spike_value)


def rk4_step(var, dt, state, max_spike_value):
    """Runge Kutta 4 method step"""
    t, val, name = state['t'], var(), var.__name__
    k1 = var.ddt(**state)
    state['t'] = t + dt / 2
    state[name] = val + (dt / 2) * k1
    k2 = var.ddt(**state)
    state[name] = val + (dt / 2) * k2
    k3 = var.ddt(**state)
    state['t'] = t + dt
    state[name] = val + dt * k3
    k4 = var.ddt(**state)
    var.value = min(var.value + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4),
                    max_spike_value)
