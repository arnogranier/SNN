from snn.network import *
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


np.random.seed(123)
tf.set_random_seed(123)
T, dt = 300, 0.3

# Size of populations
n1, n2 = 10, 10

graph = tf.Graph()
with graph.as_default():

    # Decay of population 1 : exp(-t/10)
    decay1 = lambda t: np.exp( -t / 10) ; howfar1 = 30

    # Decay of population 2 : (t/50) * exp(1-t/50)
    decay2 = lambda t: (t / 50) * np.exp(1 - (t / 50)) ; howfar2 = 100

    # Input to population 1 : 5 pour t < 15 0 sinon
    input_to_n1 = lambda t : 5*np.ones((n1,1)) if t < 15 else np.zeros((n1,1))

    # Populations
    N1 = Izhi_Nucleus(n1, label='N1', a=0.02, b=0.2, c=-65, d=8,
                      Iext=input_to_n1)
    N2 = Izhi_Nucleus(n2, label='N2', a=0.02, b=0.2, c=-65, d=8, Iext=0)

    # Connections entre les populations
    connect(N1, N2, 1, delay=30, decay=decay1, howfar=howfar1)
    connect(N1, N1, 0.5, delay=50, decay=decay2, howfar=howfar2)

    # Building tensorflow graph for this model
    nuclei = [N1, N2]
    data = build_izhi(dt, nuclei, synapse_type='simple')

# Simulate the model
vss, uss, Iss, firedss = simulate(T, dt, graph, nuclei, data)

# Plot 1 neuron of each population
print(N1.firing_rate)  # -> 6.66
plot_neuron_by_idx(T, dt, {N2:0, N1:0}, variables=['v', 'I'])
show()
