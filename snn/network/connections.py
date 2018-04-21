from .tools import *

def connect(N, M, P):
    P = treat_parameter(P, type_='connect_matrix', n1=N.n, n2=M.n)
    M.afference.append((N, P))