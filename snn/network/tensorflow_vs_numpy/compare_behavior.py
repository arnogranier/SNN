import withnp
import withtf
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from plots import raster_plot, plot_neuron_by_idx


def compare_behavior(n, input_intens, T, dt, proportion_ex_in,
                     show_neuron_idx=None):
    """Plot raster plot and the behavior of the neurons with idx in 
       show_neuron_idx for np and tf implementation """

    # Creating the function defining the external current
    Iext = lambda t: np.array(input_intens * np.ones((n, 1)), dtype=np.float32)

    # Build and simulate numpy model
    a, b, c, d, W, v, u, I, fired = withnp.build(n, proportion_ex_in, Iext)
    v_np, u_np, _, fireds_np = withnp.simulate(T, dt, a, b, c, d, W, v, u, I,
                                               fired, Iext)

    # Build and simulate tensorflow model
    graph, v, u, I, v_op, u_op, fired_op, I_op, external_input = \
     withtf.build(n, proportion_ex_in, dt, Iext)
    v_tf, u_tf, _, fireds_tf = withtf.simulate(T, dt, graph, v, u, I,
                                               external_input, Iext, v_op,
                                               u_op, fired_op, I_op)

    # Plot single neurons behavior
    if show_neuron_idx is not None:
        neuron_plot_np = plot_neuron_by_idx(T, dt, v_np, show_neuron_idx,
                                            label='whit numpy', color='b')
        neuron_plot_tf = plot_neuron_by_idx(T, dt, v_tf, show_neuron_idx,
                                            label='whit tensorflow', color='r')
    else:
        neuron_plot_np, neuron_plot_tf = None, None

    # Raster plots
    rasterfig_np = raster_plot(fireds_np, label='with numpy', color='blue')
    rasterfig_tf = raster_plot(fireds_tf, label='with tensorflow', color='red')

    return rasterfig_np, rasterfig_tf, neuron_plot_np, neuron_plot_tf


if __name__ == '__main__':

    #==========================================================================
    # PARAMETERS
    T = 200
    dt = 0.2
    proportion_ex_in = 6/10
    n = 100
    input_intens = 5
    #==========================================================================

    compare_behavior(n, input_intens, T, dt, proportion_ex_in, show_neuron_idx=0)
    plt.show()
