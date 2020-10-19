from .tools import *


def connect(N, M, P, delay=0, decay=None, howfar=0):
    """
    Connect the neuron set N to the neuron set M with connection matrix P
    by adding the tuple (N, P) to M external connection list.

    Parameters
    ----------
    N : Izhi_Nucleus
        Nucleus sending projections
    M : Izhi_Nucleus
        Nucleus receiving projections
    P : np.ndarray, tf.Tensor
        Weight matrix
    delay : int
        Synapse model parameter delay in time steps
    decay : function float (time) -> float
        Synapse model parameter decay
    howfar : int
        Synapse model parameter howfar

    """
    if decay is None:
        decay = lambda t: 0
    P = treat_parameter(P, type_='connect_matrix', n1=N.n, n2=M.n, label='P')
    M.afference.append((N, P, int(delay), decay, howfar))
