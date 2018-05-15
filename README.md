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
from snn.network import *
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

The decay function and the time window of the synapse model (see ter.pdf :
```
decay = lambda t: np.exp( - (t / 20)) ; howfar = 50
```

The external input (which could also be defined as a function of time):
```
external_input = 5
```

The parameters of the Izhikievich model, here they are set to modelize Regular Spiking neurons
