from snn.network import *
import numpy as np
import tensorflow as tf
np.random.seed(123)
tf.set_random_seed(123)

T, dt = 1000, 0.2

# Size of populations
names = ['CTX','D1','D2','Gpi','TA','TI','STN']
sizes = [2000 , 400, 400, 200 , 300, 300, 200]

# parameters
# CTX, Gpi -> Regular spiking
# D1, D2, TA, TI -> Fast spiking
# STN -> Intrinsically bursting
parameters = [
              {'a':0.02, 'b':0.2, 'c':-65, 'd':8}, #CTX
              {'a':0.1, 'b':0.2, 'c':-65, 'd':2}, #D1
              {'a':0.1, 'b':0.2, 'c':-65, 'd':2}, #D2
              {'a':0.02, 'b':0.2, 'c':-65, 'd':8}, #Gpi
              {'a':0.1, 'b':0.2, 'c':-65, 'd':2}, #TA
              {'a':0.1, 'b':0.2, 'c':-65, 'd':2}, #TI
              {'a':0.02, 'b':0.2, 'c':-55, 'd':4}, #STN
             ]

# connexion matrix
connexion_matrix = [#CTX D1  D2  Gpi TA  TI  STN
                    [0, 30,  6,   0,  0,  0, 30], #CTX
                    [0, -2, -2, -10,  0,  0,  0], #D1
                    [0, -2, -3,   0, -2, -2,  0], #D2
                    [0,  0,  0,   0,  0,  0,  0], #Gpi
                    [0, -2, -2,   0, -2, -2, -2], #TA
                    [0, -2, -2,  -2, -2, -2, -4], #TI
                    [0,  0,  0,  20, 20, 20,  0]  #STN
                   ] 

connexion_matrix[0][1] = 30

# delay matrix
delay_matrix =     [#CTX  D1    D2   Gpi   TA    TI    STN
                    [0,   10,   10,    0,    0,    0,  2.5], #CTX
                    [0,   14,   10,    7,    0,    0,    0], #D1
                    [0,   11,   13,    0,    5,    5,    0], #D2
                    [0,    0,    0,    0,    0,    0,    0], #Gpi
                    [0,    6,    6,    0,    1,    1,    4], #TA
                    [0,    6,    6,    3,    1,    1,    4], #TI
                    [0,    0,    0,  1.5,    2,    2,    0]  #STN
                   ] 

# Decay of excitatory syanapses
decay_p = lambda t: np.exp(-t / 20) ; howfar_p = 20

# Decay of inhibitory syanapses
decay_n = lambda t: (t / 50) * np.exp(1 - (t / 50)) ; howfar_n = 100

# Input to population 1 : 5 pour t < 15 0 sinon
input_to_cortex = lambda t : 7*np.random.normal(1, 3, size=(sizes[0],1))

# Weight randomization w <- (1 + nu)*w with nu gaussian process of mean 0
def randomized_w(weight):
    sign_w = np.sign(weight)
    sigma_w = np.random.normal(0, 1, size=(N.n, M.n))
    w = abs(weight) + sigma_w*abs(weight)
    w[w < 0] = 0
    return sign_w * w


graph = tf.Graph()
with graph.as_default():
    
    # Populations
    nuclei = [Izhi_Nucleus(size, label=name, **parameters,
                           Iext=0 if name != 'CTX' else input_to_cortex)
              for name, size, parameters in zip(names, sizes, parameters)]

    # Connections between populations
    for i, N in enumerate(nuclei):
        for j, M in enumerate(nuclei):
            weight = connexion_matrix[i][j]/N.n
            delay = delay_matrix[i][j]
            if weight != 0:
                connect(N, M, randomized_w(weight), delay=delay,
                        decay=decay_p if weight > 0 else decay_n,
                        howfar=howfar_p if weight > 0 else howfar_n)
    
    # Building tensorflow graph for this model
    data = build_izhi(dt, nuclei)

# Simulate the model
vss, uss, Iss, firedss = simulate(T, dt, graph, nuclei, data)
# Plot 1 neuron of each population
for nucleus in nuclei:
    raster_plot(nucleus)
plot_neuron_by_idx(T, dt, {nuclei[-2] : [0, ]})
show()
