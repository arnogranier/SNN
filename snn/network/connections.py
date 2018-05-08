from .tools import *


def connect(N, M, P):
    """Connect the neuron set N to the neuron set M with connection matrix P
    by adding the tuple (N, P) to M external connection list """
    P = treat_parameter(P, type_='connect_matrix', n1=N.n, n2=M.n)
    M.afference.append((N, P))
