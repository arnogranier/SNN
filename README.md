# SNN
[TER-MIASHS][Python 3.6-Tensorflow] Spiking Neural Network Lib

### PARTIE I

**CODE -** *A peu près fini*
1. Rentrer le modele facilement 02/02 -> ok 
2. simuler le modele 03/02 -> ok
3. phase plan and dynamics -> 05/02 -> ok
4. interactive plot phase plan with click, eventually 3d -> 07/02 -> 2/3 ok manque 3d

**WRITTING -** *En cours*
1. Intro et motivations, C.Elegans, philosophie de l'esprit 10/02 -> 1/2 ok, y'a les idées faut prendre le temps de les écrire maintenant

⋅⋅⋅Culuture humaine -> Sciences -> Maths-info-sciences de l’homme_neuro -> on pioche dans les trois -> s’inscrit dans le courant des sci co. Intérêt de la pluridisciplinarité (internet)
Une des activités de l’humanité en société est le développement de cette culture. Les motivations peuvent être différentes, à l’échelle de la société, si on considère la société comme un agent avec un but, le développement de la culture est souvent nécessaire à l’accomplissement (ou au rapprochage) de ce but, si il n’est pas le but lui-même. 
De la difficulté d’une science de l’homme, car nous sommes des hommes, objectivité et subjectivité, compréhension totale de l’Homme, est-ce un problème de complexité ou une chimère ? (http://journals.openedition.org/communication/2141)
Un abord « exact » de l’homme, fonctionnalisme et physicalisme + alternatives. Implication du fonctionnalisme, définir modélisation, modélisation de l’humain. Dans l’étude de l’individu humain, le système nerveux est considéré comme le siège du traitement de l’information et de la prise de décisions, et c’est aussi, d’après nos connaissances actuelles, la partie du corps humain la plus complexe. L’enjeu majeur de la modélisation de l’humain peut alors être située dans la modélisation du système nerveux, et principalement du système nerveux central.
Définir neurone ; depuis Ramon y Cajal il est considéré que le neurone est l’unité de base du système nerveux. Il est également globalement admis que l’activité du cerveau (qu’on considère une activité physiologique, symbolique ou sémantique) est permise et émerge de l’activité de neurones interconnectés. 
Il est donc naturel, dans une optique de modélisation de l’humain, de s’intéresser à la modélisation des neurones et des réseaux de neurones. J’aimerai dégager deux buts principaux de cette modélisation : reproduire les comportements physiologiques, dits de bas niveau (pattern d’activité électrique, notamment) et observer et étudier l’émergence de comportement dit de haut niveau (symbolique, sémantique, voire émotionnels, conscients, intelligents). Pour l’instant, nous en sommes principalement au premier but. On peut faire un parallèle avec l’IA et les réseaux de neurones formels, qui sont bien différents de l’humain mais desquels émergent des comportements de haut-niveau, parfois de manière un peu magique, mais si on étudie on comprend les mécanismes à l’œuvre dans l’émergence de ces comportements de haut-niveau (parler de Deep Dream de Google, par ex)

1. Quand faut-il privilegier la rapidite d'execution au relaisme biologique ? 11/02 (eventuellement dans l'intro, on verra)
1. Modele de Hodkin Huxley 14/02
1. Interets de réduire Hodgkin-Huxley, réduction aux modèles de Fitzugh-Nagumo, modèle de Izhiekievich 17/02
1. Modèle leaky-integrate and fire 19/02
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


