from snn.network import *
import numpy as np
import tensorflow as tf

np.random.seed(123)
tf.set_random_seed(123)
T, dt = 100, 0.3
graph = tf.Graph()
with graph.as_default():
    N1 = Izhi_Nucleus(2, a=0.03, b=0.2, c=-65, d=4, 
    				  Iext=10, W=0.7-np.random.rand(2, 2))
    N2 = Izhi_Nucleus(10, a=0.03, b=0.2, c=-65, d=4, 
    				  Iext=0, W=0.7-np.random.rand(10, 10))
    connect(N1, N2, 100 * np.random.rand(10, 2))
    nuclei = [N1, N2]
    data = build_izhi(dt, nuclei)
vss, uss, Iss, firedss = simulate(T, dt, graph, nuclei, data)
raster_plot(N2)
plot_neuron_by_idx(T, dt, N2, [5, 9])
show()
























