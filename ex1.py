import numpy as np
import tensorflow as tf
import snn.network as snn

dt = 0.1
size = 1000
decay = lambda t: np.exp(- (t / 20)) ; delay = 5 ; howfar = 50
external_input = 5
parameters = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}
W = np.random.normal(0, 1, size=(size, size))
graph = tf.Graph()
with graph.as_default():
    N = snn.Izhi_Nucleus(size, label='N', **parameters, Iext=external_input)
    snn.connect(N, N, W, delay=delay, decay=decay, howfar=howfar)
    data = snn.build_izhi(dt, [N, ], synapse_type='simple')

T = 300
snn.simulate(T, dt, graph, [N, ], data)
snn.raster_plot(N)
snn.plot_neuron_by_idx(T, dt, {N:0}, variables=['v', 'I'])
snn.show()
























