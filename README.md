# SNN
## [TER-MIASHS][Python 3.6-Tensorflow] Spiking Neural Network Lib

### _With an idea of reconciliation between mathematical and computational modelling of the nervous system and artificial intelligence_

___We know for sure (or at least with a high level of certainty) only one physical system from which generalized intelligence emerges: the nervous system. Despite that fact, artificial intelligence and mathematical and computational  modelling of the nervous system at the cellular and network level are two scientific fields that have only few interactions. Indeed, artificial intelligence mainly focuses on practical applications and solving restricted problems, whereas nervous system modelling main interests lie in health or biology studies. I think that, to tackle the problem of building a system with generalized human-like intelligence, the most efficient pathway and the only one with chances of success within the next decades is to use nervous system modelling with the intention of building a system from which cognition and intelligence emerge. And indeed we can see that when artificial intelligence try to get inspiration from the nervous system, it leads to great advances in the field, one of the latest and most known being the success story of deep learning. But I state that deep learning, with oversimplified models of neurons and synapses and abusive use of supervised learning, is not well-suited as it stands to produce a system with generalized intelligence. This work is an attempt to participate in taking the idea of artificial intelligence inspired by the nervous system a step further, with biologically plausible neurons and synapse model, brain-inspired architecture and unsupervised learning. In this work, I produce an understandable synthesis of some important features of mathematical modelling of the nervous system aimed at artificial intelligence specialists and I develop the first (in my knowledge) toolkit in python/tensorflow to simulate networks of neurons with biologically plausible features.___

<object data="https://github.com/ArnoGranier/SNN/files/2000196/ter.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/ArnoGranier/SNN/files/2000196/ter.pdf">
        Download PDF : <a href="https://github.com/ArnoGranier/SNN/files/2000196/ter.pdf">Download PDF</a> or go to ter.pdf to view it.</p>
    </embed>
</object>

__COMING SOON ON THIS README : Example of usage (already some in pdf), doc (WIP)__

__Simple example :__
_Suppose we want to simulate a network of 1000 neurons with each neurons fully-interconnected_

First we need to import snn.network : 
```
import numpy as np
import tensorflow as tf
import snn.network as snn
```

Then we need to specify some parameters :

The time step :
```
dt = 0.1
```

The size of the population : 
```
size = 1000
```

The decay function, delay and time window of the synapse model (see ter.pdf II.1.2 for more details) :
```
decay = lambda t: np.exp( - (t / 20)) ; delay = 30 ; howfar = 50
```

The external input (which could also be defined as a function of time) :
```
external_input = 5
```

The parameters of the Izhikievich model, here they are set to modelize Regular Spiking neurons (see ter.pdf I.3.3 for sets of parameters for other type of neurons) :
```
parameters = {'a': 0.02, 'b': 0.2, 'c': -65, 'd': 8}
```

The weights matrix W = (wij) where wij is the weight of the connexion from the ith neuron to the jth neuron. Here we take each weight to be a random number generated by a N(0,1) :
```
W = np.random.normal(0, 1, size=(size, size))
```

We can then build the tensorflow graph for our model using snn.network. We first build a empty tensorflow graph, then we create an Izhi_Nucleus (which is a python object that stock some data in convenient way), then we connect this Izhi_Nucleus with itself using weight matrix W and the function, delay and time window specified earlier (this just update the list of afference list of the Izhi_Nucleus), and finaly we fill the tensorflow graph with the necessary operations to simulate our model.
```{r, engine='python', count_lines}
graph = tf.Graph()
with graph.as_default():
    N = snn.Izhi_Nucleus(size, label='N', **parameters, Iext=external_input)
    snn.connect(N, N, W, delay=delay, decay=decay, howfar=howfar)
    data = snn.build_izhi(dt, [N, ], synapse_type='simple')
```
