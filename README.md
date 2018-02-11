# SNN
[TER-MIASHS][Python 3.6-Tensorflow] Spiking Neural Network Lib

### PARTIE I

**CODE -** *A peu près fini*
1. Rentrer le modele facilement -> ok 
2. simuler le modele -> ok
3. phase plan and dynimacis -> 05/02 -> ok
4. interactive plot phase plan with click, eventually 3d -> 07/02 -> 2/3 ok manque 3d

**WRITTING -** *En cours*
1. Intro et motivations, C.Elegans, philo 10/02
1. Quand faut-il privilegier la rapidite ? 11/02
1. Modele de Hodkin Huxley 14/02
1. Reduction aux modele de Fitzugh-Nagumo, modele de Izhiekievich 17/02
1. Modele leaky-integrate and fire 19/02
1. Quelques mots sur les modeles compartimentaux et augmentation de Hodkin-Huxley 20/02

Commencer le site -> 1/03
Si fini avant 1/03, s'interesser a d'autres methodes de simulations numeriques

**Pitit exemple** :
```
from brnn.single import *

v = Variable(name='v', ddt='0.04*v**2+5*v+140-u+I', init_value=-65, 
			 reset_value='c', unit='mV')
u = Variable(name='u', ddt='a*(b*v-u)', init_value=-15, reset_value='u+d')
izhi_model = Model(v, u, spike_when='v>=30', max_spike_value=30, 
        a=0.02, b=0.2, c=-65, d=8, I=lambda t : 0 if t<200 or t>800 else 5)
fig1 = izhi_model.plot(1000, 1, keep=['I', 'u', 'v'])
fig2 = izhi_model.plot(1000, 1, keep=['I', 'u', 'v'], subplotform='22')
fig3 = izhi_model.plan_phase(('v', -80, -30, 5), ('u', -30, 20, 5),
                       rescale=True)
fig4 = izhi_model.plan_phase(('v', -80, -30, 5), ('u', -30, 20, 5),
                       interactive=True, no_dynamics=True)
plt.show()
```

![alt Text](https://user-images.githubusercontent.com/27825602/36076476-e6d45882-0f5c-11e8-95d9-1e9a16462bbc.JPG)
![alt Text](https://user-images.githubusercontent.com/27825602/36076477-e7ec5936-0f5c-11e8-9830-e70f3b350b75.JPG)
![alt Text](https://user-images.githubusercontent.com/27825602/36076478-e993eede-0f5c-11e8-99e9-2dd3d4f44b07.JPG)
![alt Text](https://user-images.githubusercontent.com/27825602/36076474-e60e1bcc-0f5c-11e8-9dba-75ee33e47904.gif)





