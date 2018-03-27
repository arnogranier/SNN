import numpy as np


def build(n, exi_inhi_rate, Iext, seed=123):
    """Build parameters, weight matrix and variable init states"""

    np.random.seed(seed)

    nb_ex = round(n * exi_inhi_rate)
    nb_in = round(n * (1 - exi_inhi_rate))

    # Parameters generation as in (Izhikievich, 2003) in order to have
    # a variety of neural behavior
    re = np.array(np.random.rand(nb_ex, 1), dtype=np.float32)
    ri = np.array(np.random.rand(nb_in, 1), dtype=np.float32)
    a = np.array(np.concatenate((0.02 * np.ones((nb_ex, 1)),
                                 0.02 + 0.08 * ri)), dtype=np.float32)
    b = np.array(np.concatenate((0.2 * np.ones((nb_ex, 1)),
                                 0.25 - 0.05 * ri)), dtype=np.float32)
    c = np.array(np.concatenate((- 65 + 15 * re ** 2,
                                 - 65 * np.ones((nb_in, 1)))),
                 dtype=np.float32)
    d = np.array(np.concatenate((8 - 6 * re ** 2,
                                 2 * np.ones((nb_in, 1)))),
                 dtype=np.float32)

    # Weight matrix generation W =(wij) wich give the weight of the connexion 
    # between the neuron i and the neuron j
    # Excitator neurons have positive weights between 0 and 0.5
    # Inhibitor neurons have negative weights between -1 and 0
    W = np.random.rand(n, n)
    W[:nb_ex, :] *= 0.5
    W[nb_ex:, :] *= -1
    W = np.array(W, dtype=np.float32)

    # Initialisation state of the variables
    v = - 65 * np.ones((n, 1), dtype=np.float32)
    u = np.multiply(b, v)
    fired = np.zeros(n, dtype=np.bool)
    I = Iext(0)

    return a, b, c, d, W, v, u, I, fired


def simulate(T, dt, a, b, c, d, W, v, u, I, fired, Iext, seed=123):
    """Simulate the numpy model for T seconds with dt time step using parmeters 
       from build and where Iext is the function of time which give the 
       external current injected for each time 0<=t<T"""

    np.random.seed(seed)

    M = int(T / dt)

    # Initialization of the matrixes containg the states
    vs, us, Is, fireds = v.T, u.T, I.T, {0: fired}

    for m in range(M - 1):

        # Reset rule
        u[fired] = u[fired] + d[fired]
        v[fired] = c[fired]

        # The dynamical system
        u = u + dt * (a * (b * v - u))
        v = v + dt * (((0.04 * v) + 5) * v + 140 - u + I)

        # Get indexes of neurons that fired
        fired = np.where(v >= 30)[0]

        # Keep all neurons that fired at 30mV
        v[fired] = 30

        # Update the input, according to external input and to
        # internal influence of neurons that fired, 
        # wich is computed by summing by columns all raw of W
        # with indexes corresponding to those of neurons that fired
        I = Iext(m * dt) + np.sum(W[fired, :], axis=0, keepdims=True).T

        # Storing states
        Is = np.vstack((Is, I.T))
        vs = np.vstack((vs, v.T))
        us = np.vstack((us, u.T))
        fireds[m * dt] = fired.copy()

    return (np.array(vs, dtype=np.float32), np.array(us, dtype=np.float32),
            np.array(Is, dtype=np.float32), fireds)
