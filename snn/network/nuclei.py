from .tools import *


class Izhi_Nucleus:

    """A set of neurons with (Izhiekievich, 2003) model
        dv/dt = 0.04 * v ^ 2 + 5 * v + 140 - u + I
        du/dt = a * (b * v - u)

        if v >= 30 :
        v <- c
        u <- u + d
        """

    def __init__(self, n, a=0.03, b=0.2, c=-65, d=4,
                 Iext=5, W=0.5, label=None):

        # Stock number of neurons
        self.n = n

        # Stock parameters
        self.a = treat_parameter(a, n, label='a')
        self.b = treat_parameter(b, n, label='b')
        self.c = treat_parameter(c, n, label='c')
        self.d = treat_parameter(d, n, label='d')

        # Stock external input
        self.Iext = treat_callable(Iext, n)

        # Stock label
        self.label = label

        # Initalize external connection list
        self.afference = list()

        # Initialize historic
        self.historique = {'v': np.array([]),
                           'u': np.array([]),
                           'I': np.array([]),
                           'fired': {}, 
                           't': 0}

    # return the firing rate in Hz
    @property
    def firing_rate(self):
        fired, t = self.historique['fired'], self.historique['t']
        return (1000 / (t * self.n)) * sum([len(list_of_idx) 
                                          for list_of_idx in fired.values()])
    
