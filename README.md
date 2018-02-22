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
	
	(nldr : dans ce travail on va souvent assimiler, par facilité de rédaction, l'homme à son système nerveux, ou à son système cognitif).
	Les sciences peuvent être définies comme un moyen pour les hommes de produire ou d'utiliser de la connaissance à l'aide de moyens considérés comme rationnels, méthodiqes et objectifs. Au sein des sciences, on troue notamment les sciences formelles, dont l'objet est l'étude et la manipulation de systèmes formels, c'est-à-dire un système bien défini et autonome basé sur une axiomatique admise et à partir duquel on produit de la connaissance à l'intérieur du système en se servant de règles de déduction bien définies. On trouve également, parmis les sciences, les sciences dites "de l'Homme" ou encore "de l'esprit", "de la Culture". Cette formulation montre un interet des sciences dans l'application d'un modèle et d'une méthodologie scientifiques à un objet bien particulier : l'humain. On parle ici de science de l'homme "au singulier", c'est-à-dire qu'on parle des sciences qui s'interessent à l'humain à l'echelle de l'individu, et on parle des sciences s'interessant à l'esprit humain ou, par extension, à sa culture. L'humain est un objet particulier dans le sens où son étude parait de premier abord contredire un des critères de scientificité principal qui est celui de l'objectivité, c'est-à-dire la non-influence de l'observateur sur l'objet, en effet : dans l'étude de l'Homme l'observateur et l'objet sont de la même nature. Mais aussi et surtout, nos états intérieurs (états mentaux) nous sont propre et sont non-observable directement chez autrui, ce qui rend toute science impliquant ces états mentaux difficile.
	
	Cependant, le developpement de discipline comme la psychophysiologie ou plus généralement les neurosciences nous permet d'envisager une naturalisation de la cognition humaine, ou du moins d'une partie de la cognition humaine. C'est-à-dire qu'une partie de la cognition humain serait explicable en des termes des sciences naturelles, notamment à travers la biologie et la physique du système nerveux humain. On s'inscrit alors dans un courant naturaliste, voire physicaliste. C'est une façon de minimiser les problèmes evoqués précedemment, puisqu'une naturalisation (ou une physicalisation) de l'esprit humain permettrait alors de l'étudier de la même manière que n'importe quel autre objet des sciences naturelles (ou physiques), et on aurait de plus une explication possible des états mentaux humains par certaines propriétés de la matière. Cependant, et en accord avec une théorie fonctionnaliste, les états mentaux sont définis par leur fonction, au sein du mental ou au sein de l'organisme, la matière n'étant que la base permettant la réalisation de cette fonction. Avec cette approche physicaliste-fonctionnaliste, il est envisageable 
	
	
	(Apparté : les paradigmes d'étude de la cognition humaine) Parmis les grands paradigmes qui ont été explorés dans l'étude de l'esprit humain, on peut citer (manière non exhaustive) le behaviorisme, le cognitivisme et le connexionnisme. Le behaviorisme a tenté d'étudier l'humain en le réduisant à son comportement observable, c'est-à-dire ses interactions observables avec l'environnement. Cette approche est aujourd'hui considérée comme désuette, aussi, on ne s'attardera pas dessus. Le cognitivisme, paradigme fondateur des sciences cognitives, considère que le système nerveux humain peut (et doit) être étudié comme un système de traitement de l'information créant et manipulant des représentations symboliques et sémantiques du monde, ces représentations possédant des propriétés syntaxiques. Dans ce système cognitiviste, la pensée est comparée à une série d'application de règles (ou dit plus simplement, un calcul) sur les représentations. Le connexionnisme modélise les phénomènes mentaux comme des processus émergents de réseaux d'unités simples interconnectées. On oppose souvent cognitivisme et connexionnisme (ou pour être plus précis, computationnalisme et connexionnisme), en pensant que ce sont deux modes de pensées complétements opposés. Pourtant, les deux formulent la pensée comme un calcul sur des symboles. Les différences entre ces deux paradigmes se situent au niveau de la localisation des propriétés syntaxiques et sémantiques et dans la forme des règles de manipulation. Dans le cognitivisme, les propriétés syntaxiques et sémantiques sont attribuées aux représentations et les règles de manipulation sont algorithmiques et linéaires. Dans le paradigme connexioniste, les propriétés syntaxiques et les règles de calculs sont représentées par un réseau d'unités de bases interconnectées et dépendent entiéremment du fonctionnement de ces unités et de l'architecture du réseaux, et les règles de calculs peuvent être massivement parallèles ; tandis que les propriétés sémantiques sont attribuées au réseau en lui-même (en entier).
	
	
Un abord « exact » de l’homme, fonctionnalisme et physicalisme + alternatives. Implication du fonctionnalisme, définir modélisation, modélisation de l’humain. Dans l’étude de l’individu humain, le système nerveux est considéré comme le siège du traitement de l’information et de la prise de décisions, et c’est aussi, d’après nos connaissances actuelles, la partie du corps humain la plus complexe. L’enjeu majeur de la modélisation de l’humain peut alors être située dans la modélisation du système nerveux, et principalement du système nerveux central.   
Définir neurone ; depuis Ramon y Cajal il est considéré que le neurone est l’unité de base du système nerveux. Il est également globalement admis que l’activité du cerveau (qu’on considère une activité physiologique, symbolique ou sémantique) est permise et émerge de l’activité de neurones interconnectés.  
Il est donc naturel, dans une optique de modélisation de l’humain, de s’intéresser à la modélisation des neurones et des réseaux de neurones. J’aimerai dégager deux buts principaux de cette modélisation : reproduire les comportements physiologiques, dits de bas niveau (pattern d’activité électrique, notamment) et observer et étudier l’émergence de comportement dit de haut niveau (symbolique, sémantique, voire émotionnels, conscients, intelligents). Pour l’instant, nous en sommes principalement au premier but. On peut faire un parallèle avec l’IA et les réseaux de neurones formels, qui sont bien différents de l’humain mais desquels émergent des comportements de haut-niveau, parfois de manière un peu magique, mais si on étudie on comprend les mécanismes à l’œuvre dans l’émergence de ces comportements de haut-niveau (parler de Deep Dream de Google, par ex)  
Création d'un outil de mes propres outils de simulation, interet (parler de la poursuite d'étude, bonne facon d'apprendre les différentes méthodes en simulation, manque d'environnements de création de réseaux de spiking neurons en python (brian mais bon,et tensorflow > all)).

1. Quelques rappels de neurobiologie
	Cette partie sera concise et aura pour but de rappeller quelques notions de neurobiologie necessaire à la compréhension de la suite, sans en faire trop. On suppose que le lecteur est déjà familié avec les notions fondamentales de la neurobiologie, si ce n'est pas le cas, on renvoie à Principles of Neural Science de Eric Kandel.  
Le neurone est une cellule capable de recevoir et transmettre de l'information sous forme electro-chimique. On peut décomposer schématiquement les différentes étapes de la reception et transmition de l'information in vivo dans un neurone par :
	1. Reception de neurotransmetteurs et ouverture des canaux chimio-dépendants
	2. Excitation electrique locale du neurone dû à l'ouverture des canaux chimio-dépedants
	3. Lorsque l'excitation locale depasse un certain seuil, création d'un potentiel d'action
	4. Transmition du potentiel d'action à travers l'axone
	5. Libération de neurotransmetteur dans la fente synaptique dû à l'arrivée du potentiel d'action dans le bouton synaptique 
	6. Repeter 1. pour le neurone post-synaptique

	Lorsqu'on souhaite étudier les propriétés d'excitation d'un neurone en laboratoire, on va généralement provoquer l'excitation du neurone en injectant directement un courant electrique dans le neurone, et on va s'interesser à la production de potentiels d'action en fonction des propriétés du courant injecté, notamment de son intensité (technique de patch-clamp). 

	Dans cette idée d'étude de la production de potentiel d'action en fonction des propriétés d'un courant injecté directement dans le neurone, on ne décrira pas ici les mécanismes à l'oeuvre dans la synapse.

	Le concept fondamental de neurobiologie en lien avec cette partie est celui de la création du potentiel d'action. On rappellera ici succintement les mecanismes neurobiologiques à l'oeuvre. On peut décomposer la génération d'un potentiel d'action en 5 phases :
	1. Dépolarisation faible : ouverture de certain canaux sodiques, entrée des ions Na2+ dans le milieu intracellulaire ;
	2. Dépolarisation forte suite au dépassement de seuil : Lorsqu'un certain seuil de potentiel electrique est atteint (le potentiel de seuil), la membrane va subir une dépolarisation forte, allant jusqu'à un inversement de polarité où le potentiel de la membrane est d'environ 40 mV. Cette dépolarisation est due à l'ouverture massive de canaux sodiques. Une fois le changement de polarité effectué, l'inversion du gradient electrochimique va ralentir l'entrée des ions Na2+ dans la cellule ;
	3. Repolarisation : L'ouverture des canaux potassiques et l'inactivation des canaux sodiques entraine la sortie massive d'ions K+ et un arret de l'entrée des ions Na2+;
	4. Hyperpolarisation : En continuité de la repolarisation, on observe que le potentiel membranaire ne revient pas directement au potentiel de repos, mais passe sous le potentiel de repos pendant un certain temps que l'on appelle la période refractaire. Cela est du au fait que les canaux potassiques restent ouverts plus longtemps que les canaux sodiques, on a donc une sortie d'ions K+ plus importante que necessaire pour revenir au potentiel de repos;
	5. Retour au potentiel de repos : Le retour au potentiel de repos est assuré par la pompe Na2+/K+ ATPase.
	
	![alt Text](https://user-images.githubusercontent.com/27825602/36450932-b77c1b44-168f-11e8-975d-c6203f21f177.jpg)  
	FIG1 - Potentiel de la membrane en fonction du temps lors de la production d'un potentiel d'action

1. Modele de Hodgkin Huxley 14/02
	L'approche de Hodgkin et Huxley sur la question de la modélisation de neurones est une approche qui possède une grande "clarté physiologique", dans le sens où chaque composante du modèle représente une réalité biologique ou electrique descriptible dans les termes de la neurobiologie. On peut ainsi attribuer au modèle de Hodgkin-Huxley une certaine cohérence et validité par rapport aux sciences naturelles (notamment neurobiologie encore une fois). Mais voyons ça plus en détails ..
	1. Les equations des canaux ioniques
	on suppose que y'en a que deux, 4 compartiment, des formules
	2. L'équation du potentiel membrane
	Représentation par circuit RC, Interprétation et appliquation de la loi de Kirchhoff pour trouver l'équation
	3. Simulation avec SNN.single et explication des dynamiques à partir de m,n,h (voir Wash)
	

1. Interets de réduire Hodgkin-Huxley (Quand faut-il privilegier la rapidite d'execution au relaisme biologique ?), réduction aux modèles de Fitzugh-Nagumo, modèle de Izhiekievich 17/02
	Expliquer le compromis entre réalisme biologique et rapidité d'execution, modèle phéno et modèle physio. Le but principal de ce TER est d'étudier et de produire un outil poural simulation de RESEAUX de neurones de grande taille, ainsi dans cette partie qui porte sur la modélisation d'un seul neurone, on a tout interet à se concentrer sur les modèles qui sont des simplifications de HH.
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


