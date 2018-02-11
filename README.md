# SNN
[TER-MIASHS][Python 3.6-Tensorflow] Spiking Neural Network Lib

### PARTIE I

**CODE -** *A peu près fini*
Rentrer le modele facilement -> ok
simuler le modele -> ok
phase plan and dynimacis -> 05/02 -> ok
interactive plot phase plan with click, eventually 3d -> 07/02 -> 2/3 ok manque 3d

**WRITTING -** *En cours*
Intro et motivations, C.Elegans, philo 10/02
Quand faut-il privilegier la rapidite ? 11/02
Modele de Hodkin Huxley 14/02
Reduction aux modele de Fitzugh-Nagumo, modele de Izhiekievich 17/02
Modele leaky-integrate and fire 19/02
Quelques mots sur les modeles compartimentaux et augmentation de Hodkin-Huxley 20/02

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





