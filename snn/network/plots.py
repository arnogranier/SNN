import matplotlib.pyplot as plt
import numpy as np
from .tools import is_iterable


def raster_plot(nucleus, label=None, **kwargs):
    """Raster plot fireds data of nucleus, return the matplotlib figure.

    Parameters
    ----------
    nucleus : Izhi_Nucleus
        Nucleus containing the data that we want
    label : str
        Label

    Returns
    -------
    matplotlib.pyplot.figure

    """

    # Get fireds data
    fireds = nucleus.historique['fired']

    # Build x-axis and y-axis data by looping through fireds
    xdata, ydata = [], []
    for x, fires in fireds.items():
        for y in fires:
            xdata.append(x)
            ydata.append(y)

    # Matplotlib stuff
    fig = plt.figure()
    plt.plot(xdata, ydata, '|', **kwargs)
    plt.ylabel('Neuron indices')
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


def plot_neuron_by_idx(T, dt, dict_nucleus_idxs, names=None,
                       variables=['v', ], **kwargs):
    """Plot the activity of the neuron at indices idxs of nucleus.

    Parameters
    ----------
    T : float
        Elapsed time
    dt : float
        Time step
    dict_nucleus_idxs : dict
        Dict of the form {Izhi_Nucleus: list of idxs} representing the neurons
        from which want to plot
    names : list of str
        List of names for the different neurons in dict_nucleus_idxs
    variables : list of str
        Names of the variables to plot

    Returns
    -------
    matplotlib.pyplot.figure

    """

    for (nucleus, idxs) in dict_nucleus_idxs.items():
        if not is_iterable(idxs):
            dict_nucleus_idxs[nucleus] = [idxs, ]

    # We build one figure for each variable to plot
    figs = []
    for figidx, varname in enumerate(variables):
        fig = plt.figure()

        # Plot data
        for (nucleus, idxs) in dict_nucleus_idxs.items():
            for idx in idxs:
                plt.plot(np.linspace(0, T, int(T/dt)),
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
