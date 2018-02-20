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

	Culture humaine -> Sciences -> Maths-info-sciences de l’homme_neuro -> on pioche dans les trois -> s’inscrit dans le courant des sci co. Intérêt de la pluridisciplinarité (internet)  
Une des activités de l’humanité en société est le développement de cette culture. Les motivations peuvent être différentes, à l’échelle de la société, si on considère la société comme un agent avec un but, le développement de la culture est souvent nécessaire à l’accomplissement (ou au rapprochage) de ce but, si il n’est pas le but lui-même. A l'echelle de l'individu, la survie étant facilité dans notre environnement actuel, des questions et des envies nouvelles et basées sur la culture humaine antérieure apapraissent, et la recherche de réponses à ces questions ou la recherche de l'accomplissement de ces envies peut en retour mener à la modification de la Culture.    
De la difficulté d’une science de l’homme, car nous sommes des hommes, objectivité et subjectivité, compréhension totale de l’Homme, est-ce un problème de complexité ou une chimère ? (http://journals.openedition.org/communication/2141)  
Un abord « exact » de l’homme, fonctionnalisme et physicalisme + alternatives. Implication du fonctionnalisme, définir modélisation, modélisation de l’humain. Dans l’étude de l’individu humain, le système nerveux est considéré comme le siège du traitement de l’information et de la prise de décisions, et c’est aussi, d’après nos connaissances actuelles, la partie du corps humain la plus complexe. L’enjeu majeur de la modélisation de l’humain peut alors être située dans la modélisation du système nerveux, et principalement du système nerveux central.   
Définir neurone ; depuis Ramon y Cajal il est considéré que le neurone est l’unité de base du système nerveux. Il est également globalement admis que l’activité du cerveau (qu’on considère une activité physiologique, symbolique ou sémantique) est permise et émerge de l’activité de neurones interconnectés.  
Il est donc naturel, dans une optique de modélisation de l’humain, de s’intéresser à la modélisation des neurones et des réseaux de neurones. J’aimerai dégager deux buts principaux de cette modélisation : reproduire les comportements physiologiques, dits de bas niveau (pattern d’activité électrique, notamment) et observer et étudier l’émergence de comportement dit de haut niveau (symbolique, sémantique, voire émotionnels, conscients, intelligents). Pour l’instant, nous en sommes principalement au premier but. On peut faire un parallèle avec l’IA et les réseaux de neurones formels, qui sont bien différents de l’humain mais desquels émergent des comportements de haut-niveau, parfois de manière un peu magique, mais si on étudie on comprend les mécanismes à l’œuvre dans l’émergence de ces comportements de haut-niveau (parler de Deep Dream de Google, par ex)  

1. Quelques rappels de neurobiologie
	Cette partie sera concise puisqu'on suppose que le lecteur est déjà familié avec les notions fondamentales de la neurobiologie. Si ce n'est pas le cas, on renvoie à Principles of Neural Science de Eric Kandel.  
Le neurone est une cellule capable de recevoir et transmettre de l'information sous forme electro-chimique. On peut décomposer schématiquement les différentes étapes de la reception et transmition de l'information in vivo dans un neurone par :
	1. Reception de neurotransmetteurs et ouverture des canaux chimio-dépendants
	2. Excitation electrique locale du neurone dû à l'ouverture des canaux chimio-dépedants
	3. Lorsque l'excitation locale depasse un certain seuil, création d'un potentiel d'action
	4. Transmition du potentiel d'action à travers l'axone
	5. Libération de neurotransmetteur dans la fente synaptique dû à l'arrivée du potentiel d'action dans le bouton synaptique 
	6. Repeter 1. pour le neurone post-synaptique

	Lorsqu'on souhaite étudier les propriétés d'excitation d'un neurone en laboratoire, on va généralement provoquer l'excitation du neurone en injectant directement un courant electrique dans le neurone, et on va s'interesser à la production de potentiels d'action en fonction des propriétés du courant injecté, notamment de son intensité (technique de patch-clamp). 

	Dans cette idée d'étude de la production de potentiel d'action en fonction des propriétés d'un courant injecté directement dans le neurone, on ne décrira pas ici les mécanismes à l'oeuvre dans la synapse.

	Le concept principal de neurobiologie à comprendre ici est la création du potentiel d'action, dont la base est une série d'échange d'ions entre le milieu extracellulaire et le milieu intracellulaire. Cet echange est permis par les canaux ioniques placés le long de la membrane plasmique, qui la rende permeable à certains ions dans certaines conditions. Ces canaux ioniques peuvent être dans différents états : actifs ou actifs, ouverts ou fermés. Les canaux ioniques dont on parle ici sont des canaux voltage-dépendant, c'est-à-dire que leur état dépend de la tension electrique locale. Ces canaux ioniques sont également ion-specifique, dans le sens où ils ne rendent la membrane permeable qu'à un certain type d'ion ; ainsi on a des canaux ioniques pour les ions K+, des canaux ioniques pour les ions Na2+, etc... Les ions ne peuvent circuler à travers la membrane que lorsque le canal est actif et ouvert.  
	Dès lors que la tension electrique locale dépasse un certain seuil (soit, comme dit précedemment, grâce à l'ouverture des canaux chimio-dépendants, soit par une injection direct de courant dans le neurone), on va observer une série d'ouvertures et de fermetures de certains canaux ioniques qui vont être à l'origine du potentiel d'action, qui peuvent être résumées par :
	1.

1. Modele de Hodgkin Huxley 14/02
	On sait, de la neurobiologie, que les canaux ioniques peuvent prendreMode de fonctionnement des canaux ioniques (actif/inactif, ouvert/fermé), équations associées aux canaux ioniques, flux potassique et sodique. Modélisation par cricuit RC et explication de l'équation complète avec la loi de Kirchhoff. Estimation des paramètres. Explication des dynamiques à partir de m,n,h (voir Washington). Simulation avec SNN.single 


1. Interets de réduire Hodgkin-Huxley (Quand faut-il privilegier la rapidite d'execution au relaisme biologique ?), réduction aux modèles de Fitzugh-Nagumo, modèle de Izhiekievich 17/02
	Expliquer le compromis entre réalisme biologique et rapidité d'execution. Le but principal de ce TER est d'étudier et de produire un outil poural simulation de RESEAUX de neurones de grande taille, ainsi dans cette partie qui porte sur la modélisation d'un seul neurone, on a tout interet à se concentrer sur les modèles qui sont des simplifications de HH.
Explication de la réduction au modèles de FHN, simulations avec SNN.single, puis explications de FHN -> Izhi, encore une fois simulation.


1. Modèle leaky-integrate and fire 19/02
	Encore plus simple : le modèle leaky-integrate and fire, présentation du modèle, explications, simulations.


1. Quelques mots sur les modeles compartimentaux et augmentation de Hodkin-Huxley 20/02
	Principe (approximation des différentes composantes par des cylindres) et équation du cable. Interets ( biologiquement + réaliste). Techniques de réduction du nombre de compartiments avec same beahvior. 



Commencer le site -> 1/03
Si fini avant 1/03, s'interesser a d'autres methodes de simulations numeriques (r4k done)

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


