from .tools import *

class Izhi_Nucleus:
    def __init__(self, n, a=0.03, b=0.2, c=-65, d=4, 
                 Iext=5, W=0.5, label=None):
        self.n = n
        self.a = treat_parameter(a, n)
        self.b = treat_parameter(b, n)
        self.c = treat_parameter(c, n)
        self.d = treat_parameter(d, n)
        self.W = treat_parameter(W, type_='connect_matrix', n1=n, n2=n)
        self.Iext = treat_callable(Iext, n)
        self.label = label
        self.afference = list()
        self.historique = {'v':np.array([]), 
                           'u':np.array([]), 
                           'I':np.array([]), 
                           'fired':{}}