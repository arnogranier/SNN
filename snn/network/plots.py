import matplotlib.pyplot as plt
import numpy as np
from .tools import is_iterable


def raster_plot(nucleus, label=None, **kwargs):
    """Raster plot fireds data of nucleus, return the matplotlib figure"""

    # Get fireds data
    fireds = nucleus.historique['fired']

    # Build x-axis and y-axis data by looping through fireds
    xdata, ydata = [], []
    for x, fires in fireds.items():
        for y in fires:
            xdata.append(x)
            ydata.append(y)

    # Matplotlib
    fig = plt.figure()
    plt.plot(xdata, ydata, '|', **kwargs)
    plt.ylabel('Neuron indexes')
    plt.xlabel('time')
    if label is None:
        plt.title('Raster plot')
    else:
        plt.title('Raster plot %s' % label)

    return fig


def plot_neuron_by_idx(T, dt, nucleus, idxs, names=None,
                       variables=['v', ], **kwargs):
    """Plot the activity of the neuron at indexs idxs of nucleus"""

    if not is_iterable(idxs):
        idxs = [idxs, ]

    # There is as much figure as variables to plot
    figs = []
    for figidx, varname in enumerate(variables):
        fig = plt.figure()

        # Plot data
        for idx in idxs:
            plt.plot(np.linspace(0, T, T/dt),
                     nucleus.historique[varname][:, idx],
                     label='Neuron %s' % idx, **kwargs)

        # Handle names
        if names is None:
            plt.title('Individual neurons plot for variable %s' % varname)
        else:
            plt.title(names[figidx])

        plt.legend()
        figs.append(fig)
    return figs


def show():
    """matplotlib.show"""
    plt.show()


def savefig(path):
    """matplotlib.savefig"""
    plt.savefig(path)
