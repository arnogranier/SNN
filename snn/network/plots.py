import matplotlib.pyplot as plt
import numpy as np

def raster_plot(fireds, label=None, **kwargs):
    fig = plt.figure()
    xdata, ydata = [], []
    for x, fires in fireds.items():
        for y in fires:
            xdata.append(x)
            ydata.append(y)
    plt.plot(xdata, ydata, '|', **kwargs)
    plt.ylabel('Neuron indexes') ; plt.xlabel('time')
    if label is None : plt.title('Raster plot')
    else: plt.title('Raster plot %s' % label)
    return fig

def plot_neuron_by_idx(T, dt, v, idxs, label=None, **kwargs):
    try:iter(idxs)
    except:idxs = [idxs, ]
    fig = plt.figure()
    for idx in idxs:
        plt.plot(np.linspace(0, T, T/dt), v[:, idx], 
                 label='Neuron %s' % idx, **kwargs)
    plt.legend()
    if label is None : plt.title('Individual neurons plot')
    else: plt.title('Individual neurons plot %s' % label)
    return fig
    
