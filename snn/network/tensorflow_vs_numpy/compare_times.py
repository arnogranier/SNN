import time
import withnp
import withtf
import numpy as np
import matplotlib.pyplot as plt


def get_times(T, dt, ex_in_rate, ns, input_intensity, stop_between=0):
    """Return computation times for numpy and tensorflow models for
       networks with n neuron for n in ns"""

    # Storage vectors for computation times
    tf_build_times = np.empty((len(ns), 1), dtype=np.float32)
    np_build_times = np.empty((len(ns), 1), dtype=np.float32)
    tf_simul_times = np.empty((len(ns), 1), dtype=np.float32)
    np_simul_times = np.empty((len(ns), 1), dtype=np.float32)

    for x, n in enumerate(ns):

        print('Processing for %s synapses' % n**2)

        # Creating the function defining the external current
        Iext = lambda t: np.array(input_intensity*np.ones((n, 1)),
                                  dtype=np.float32)

        time.sleep(stop_between)

        # Building numpy model
        start_t = time.time()
        a, b, c, d, W, v, u, I, fired = withnp.build(n, ex_in_rate, Iext)
        np_build_t = time.time() - start_t

        # Simulating numpy model
        start_t = time.time()
        _, _, _, fireds_np = withnp.simulate(T, dt, a, b, c, d, W, v, u, I,
                                             fired, Iext)
        np_simul_t = time.time() - start_t

        print('Numpy : %s' % np_simul_t)

        time.sleep(stop_between)

        # Building tensorflow model
        start_t = time.time()
        graph, v, u, I, v_op, u_op, fired_op, I_op, external_input = \
         withtf.build(n, ex_in_rate, dt, Iext)
        tf_build_t = time.time() - start_t

        # Simulating tensorflow model
        start_t = time.time()
        _, _, _, fireds_tf = withtf.simulate(T, dt, graph, v, u, I,
                                             external_input, Iext, v_op,
                                             u_op, fired_op, I_op)
        tf_simul_t = time.time() - start_t

        print('Tensorflow : %s\n' % tf_simul_t)

        # Store computation times
        tf_build_times[x, 0] = tf_build_t
        np_build_times[x, 0] = np_build_t
        tf_simul_times[x, 0] = tf_simul_t
        np_simul_times[x, 0] = np_simul_t

    return tf_build_times, np_build_times, tf_simul_times, np_simul_times


if __name__ == '__main__':
    #==========================================================================
    # PARAMETERS
    T, dt = 400, 0.2
    ex_in_rate = 6/10
    ns = [10, 100, 1000, 2500, 5000, 7500, 10000]
    input_intensity = 5
    #==========================================================================

    def stuff_on_all_graph():
        """Things that will alway be used for those plots"""
        plt.gca().grid()
        plt.xscale('log')
        plt.xlabel('Number of synapses')

    # Get computation times
    tf_build_times, np_build_times, tf_simul_times, np_simul_times = \
     get_times(T, dt, ex_in_rate, ns, input_intensity, stop_between=0)

    # Plotting ratio of computation times against the number of synapses
    plt.plot([n**2 for n in ns], np_simul_times / tf_simul_times)
    plt.ylabel('Time taken by numpy / time taken by tensorflow')
    plt.title('Simulation times : numpy VS Tensorflow')
    plt.yscale('log')   # log-scale for the y-axis
    stuff_on_both_graph()
    plt.tight_layout()
    plt.show()
