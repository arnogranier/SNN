import matplotlib.pyplot as plt
import time
import withnp
import withtf
import numpy as np 
import tensorflow as tf

def raster_plot(fireds, withlabel=None):
    fig = plt.figure()
    xdata, ydata = [], []
    for x, fires in fireds.items():
        for y in fires:
            xdata.append(x)
            ydata.append(y)
    plt.plot(xdata, ydata, '|', color='black' )
    plt.ylabel('Neuron indexes') ; plt.xlabel('time')
    if withlabel is None : plt.title('Raster plot')
    else: plt.title('Raster plot with %s' % withlabel)
    return fig

def compare_time(T, dt, proportion_ex_in, ns, inputs_intensity):
    build_times_np_minus_tf = np.empty((len(ns), 1))
    simul_times_np_minus_tf = np.empty((len(ns), len(inputs_intensity)))
    for x, n in enumerate(ns):
        for y, intens in enumerate(inputs_intensity):
            Iext = lambda t : np.array(intens*np.ones((n, 1)), dtype=np.float32)

            time.sleep(10)
            
            start_t = time.time()
            a, b, c, d, W, v, u, I, fired = withnp.build(n, proportion_ex_in, Iext)
            np_build_t = time.time() - start_t

            start_t = time.time()
            _, _, _, fireds_np = withnp.simulate(T, dt, a, b, c, d, W, v, u, I, fired, Iext)
            np_simul_t = time.time() - start_t

            time.sleep(10)
            
            start_t = time.time()
            graph, v, u, I, v_op, u_op, fired_op, I_op, external_input = withtf.build(n, proportion_ex_in, dt, Iext)
            tf_build_t = time.time() - start_t

            start_t = time.time()
            _, _, _, fireds_tf = withtf.simulate(T, dt, graph, v, u, I, external_input, Iext, v_op, u_op, fired_op, I_op)
            tf_simul_t = time.time() - start_t

            build_times_np_minus_tf[x, 0] = np_build_t - tf_build_t
            simul_times_np_minus_tf[x, y] = np_simul_t - tf_simul_t
            tf.reset_default_graph()
            print(n, intens)
            
    figsimul = plt.figure()
    plt.imshow(simul_times_np_minus_tf)
    plt.title('Simulation time elapsed : np-tf')
    plt.xlabel('Input intensity')
    plt.ylabel('number of synapses')
    plt.gca().set_xticks(range(len(inputs_intensity)))
    plt.gca().set_xticklabels(inputs_intensity)
    plt.gca().set_yticks(range(len(ns)))
    plt.gca().set_yticklabels([n**2 for n in ns])
    plt.colorbar()

    figbuild = plt.figure()
    plt.imshow(build_times_np_minus_tf)
    plt.title('Building time elapsed : np-tf')
    plt.ylabel('number of synapses')
    plt.gca().set_yticks(range(len(ns)))
    plt.gca().set_yticklabels([n**2 for n in ns])
    plt.colorbar()

    plt.tight_layout()
    return figsimul, figbuild

def compare_behavior(n, Iext, T, dt, proportion_ex_in, show_neuron_idx=0):
    a, b, c, d, W, v, u, I, fired = withnp.build(n, proportion_ex_in, Iext)
    v_np, u_np, _, fireds_np = withnp.simulate(T, dt, a, b, c, d, W, v, u, I, fired, Iext)
    graph, v, u, I, v_op, u_op, fired_op, I_op, external_input = withtf.build(n, proportion_ex_in, dt, Iext)
    v_tf, u_tf, _, fireds_tf = withtf.simulate(T, dt, graph, v, u, I, external_input, Iext, v_op, u_op, fired_op, I_op)

    neuronfig = plt.figure()
    plt.plot(np.linspace(0, T, T/dt), v_np[:, show_neuron_idx], label='with numpy')
    plt.plot(np.linspace(0, T, T/dt), v_tf[:, show_neuron_idx], label='with tensorflow')
    plt.legend()
    raster1fig = raster_plot(fireds_np, withlabel='numpy')
    raster2fig = raster_plot(fireds_tf, withlabel='tensorflow')

    return neuronfig, raster1fig, raster2fig
    
if __name__ == '__main__':
    T, dt = 400, 0.2
    proportion_ex_in = 8/10
    ns = [ int(1.5*10**4), 10**1, 10**2, 10**3, int(2.5*10**3), 5*10**3, int(7.5*10**3), 10**4,]
    inputs_intensity = range(2,13,2)
    fig1, fig2 = compare_time(T, dt, proportion_ex_in, ns, inputs_intensity)

##    T, dt = 100, 0.2
##    proportion_ex_in = 8/10
##    n = 10**2
##    Iext = lambda t : np.array(10*np.ones((10**2, 1)), dtype=np.float32)
##    compare_behavior(n, Iext, T, dt, proportion_ex_in)
##    plt.show()
    

