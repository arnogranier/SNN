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
    if label is None and nucleus.label is None:
        plt.title('Raster plot')
    elif label is not None and nucleus.label is None:
        plt.title('Raster plot (%s)' % label)
    elif label is None and nucleus.label is not None:
        plt.title('Raster plot of %s' % nucleus.label)
    else:
        plt.title('Raster plot of %s (%s)' % (nucleus.label, label))

    return fig


def plot_neuron_by_idx(T, dt, dico_nucleus_idxs, names=None,
                       variables=['v', ], **kwargs):
    """Plot the activity of the neuron at indexs idxs of nucleus"""

    for (nucleus, idxs) in dico_nucleus_idxs.items():
        if not is_iterable(idxs):
            dico_nucleus_idxs[nucleus] = [idxs, ]

    # There is as much figure as variables to plot
    figs = []
    for figidx, varname in enumerate(variables):
        fig = plt.figure()

        # Plot data
        for (nucleus, idxs) in dico_nucleus_idxs.items():
            for idx in idxs:
                plt.plot(np.linspace(0, T, T/dt),
                         nucleus.historique[varname][:, idx],
                         label='Neuron %s - %s' % (nucleus.label, idx), **kwargs)

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
