# SNN
[TER-MIASHS][Python 3.6-Tensorflow] Spiking Neural Network Lib
Dans une idée de rapprochement entre la modélisation du système nerveux et l'intelligence artificielle

### PARTIE I

**CODE -** *A peu près fini*
1. Rentrer le modele facilement 02/02 -> ok 
2. simuler le modele 03/02 -> ok
3. phase plan and dynamics -> 05/02 -> ok
4. interactive plot phase plan with click, eventually 3d -> 07/02 -> 2/3 ok manque 3d

**WRITTING -** *En cours*
1. Intro -> OK (putain enfin, si je puis me permettre -_-)
	
	(nldr : dans ce travail on va souvent assimiler, par facilité de rédaction, l'homme à son système nerveux, ou à son système cognitif. On essaiera d'employer le terme "système nerveux" plutot que "cerveau" ou "encephale", pour rester général et conserver l'importance du système nerveux périphérique.).
	Les sciences peuvent être définies comme un moyen pour les hommes de produire ou d'utiliser de la connaissance à l'aide de moyens considérés comme rationnels, méthodiques et objectifs. Au sein des sciences, on troue notamment les sciences formelles, dont l'objet est l'étude et la manipulation de systèmes formels, c'est-à-dire un système bien défini et autonome basé sur une axiomatique admise et à partir duquel on produit de la connaissance à l'intérieur du système en se servant de règles de déduction bien définies. On trouve également, parmis les sciences, les sciences dites "de l'Homme" ou encore "de l'esprit", "de la Culture". Cette formulation montre un interet des sciences dans l'application d'un modèle et d'une méthodologie scientifiques à un objet bien particulier : l'humain. On va parler ici de science de l'homme "au singulier", c'est-à-dire qu'on parle des sciences qui s'interessent à l'humain à l'echelle de l'individu, des sciences s'interessant à l'esprit humain. L'humain est un objet particulier dans le sens où son étude parait de premier abord contredire un des critères de scientificité principal qui est celui de l'objectivité, c'est-à-dire la non-influence de l'observateur sur l'objet, en effet : dans l'étude de l'Homme l'observateur et l'objet sont de la même nature. Mais aussi et surtout, nos états intérieurs (états mentaux) nous sont propre et sont non-observable directement chez autrui, ce qui rend toute science impliquant ces états mentaux difficile.
	
	Cependant, le developpement de discipline comme la psychophysiologie ou plus généralement les neurosciences nous permet d'envisager une naturalisation de la cognition humaine (ou du moins d'une partie de la cognition humaine). C'est-à-dire que la cognition humaine serait explicable en des termes des sciences naturelles, notamment à travers la biologie et la physique du système nerveux humain. On s'inscrit alors dans un courant naturaliste, voire physicaliste. C'est une façon de minimiser les problèmes evoqués précedemment, puisqu'une naturalisation (ou une physicalisation) de l'esprit humain permettrait alors de l'étudier de la même manière que n'importe quel autre objet des sciences naturelles (ou physiques), et on aurait de plus une explication possible des états mentaux humains par certaines propriétés de la matière. De plus, et en accord avec une théorie fonctionnaliste, on va plutot définir les états mentaux par leur fonction, au sein du mental ou au sein de l'organisme, plutot que par leur substrat physique, la matière n'étant que la base permettant la réalisation de cette fonction. Avec cette approche physicaliste-fonctionnaliste, __il est donc théoriquement envisageable de reproduire l'esprit humain dans une machine, pourvu que l'on reproduise toute les propriétés physiques du système nerveux humain.__ Le physicalisme nous permettant d'attribuer entierement l'esprit humain aux propriétés physiques du système nerveux humain, et le fonctionalisme nous permettant de nous affranchir d'une incarnation forcément dans le système nerveux pour nous étendre à une incarnation possible dans tout système _fonctionnant comme_ le système nerveux. 
	
	Mais qu'est-ce que ça veut dire _fonctionner comme_ le système nerveux ? Pour répondre à cete question, il nous faut nous tourner vers les sciences naturelles : neurosciences, biologie, physique, etc ... qui étudient les propriétés biologiques, chimiques et physiques du cerveau. Ces sciences nous apprennent que le système nerveux humain est un système extremement complexe, et nos connaissances sur les propriétés biologiques, chimiques et physiques de ce système sont loins d'être complètes. Si l'on veut tenter de résumer le fonctionnement du système nerveux en quelques mots, on aurait tendance à dire qu'il s'agit d'un réseaux organisé et adaptatif d'unités de base connectées entre elles, dont le but et de recevoir, analyser et transmettre de l'information. Cette réduction est très schématique mais semble pourtant contenir l'essence du fonctionnement du système nerveux humain. L'unité de base de ce système est le neurone, qui est une cellule capable de recevoir et de propager de l'information sous forme electro-chimique (l'influx nerveux). Les connexions entre les neurones sont appellés synapses, ce sont des zones où l'information est transmises d'un neurone à l'autre de manière chimique, et il est également globalement admit que c'est au sein des synapses que prennent place les propriétés d'adaptation du réseau.
	
	Si l'on reste dans l'approche physicaliste-fonctionnaliste, il semble donc naturel de vouloir tente de "reproduire" le fonctionnement du système nerveux humain dans _autre chose que l'humain_, et le meilleur candidat pour cet _autre chose_ semble être l'ordinateur. Cependant, il est ici important d'être lucide sur le sens du mot "reproduire" dans cette phrase : d'une part, comme nous l'avons dit, nous sommes loin d'avoir une connaissance exhaustive des propriétés des neurones et des synapses, et de plus le système nerveux ne se résume pas en réalité qu'aux neurones et synapses (il faudrait prendre en compte les cellules gliales, l'impact des hormones, reproduire le fonctionnement de toutes les afférences aux systèmes nerveux, comme les recepteurs cutanés, etc ..) ; et d'autre part, en supposant une connaissance exhaustive du système nerveux, la reproduction exacte de son fonctionnement _in silico_ ne serait peut-être pas si aisée, notamment car le substrat biologique permet peut-être des fonctions difficilement reproductible dans un substrat electronique. C'est pour cela que plutot que de parler de "reproductioné du fonctionnement du système nerveux, on parlera plutot de modélisation du système nerveux, de modélisation de neurones et de synapses, dans le sens où on selectionne les propriétés du système nerveux qui nous semblent les plus importantes dans son fonctionnement et où on essaye de les rendre intelligiebles, pour la machine grâce à une formalisation mathématique et à des programmes permettant de simuler le comportement des modèles de neurone et de réseaux de neurones, et pour l'homme à l'aide de graphiques, de données bien choisies et d'analyse mathématique des modèles (lorsque cela est possible).
	
	J'aimerai dégager deux grands axes dans l'activité de la modélisation mathématique et informatique du système nerveux :
	1. Reproduire les propriétés physique et biologiques du système nerveux (ie ici des neurones et réseaux de neurones) 
	2. Faire emerger des propriétés cognitives à partir de modèles du système nerveux et observer, analyser et comprendre cette emergence.

	Un certain avancement dans le premier axe étant bien evidemment necessaire à l'accomplissement du second.

	De la première proposition (1.) on peut dégager deux approches :  
		a. __Reproduire le plus fidélement possible un neurone ou un réseau de neurone sans se soucier de la complexité du modèle__, ce qui donne un modèle plus précis, plus proche de la réalité, mais difficilement manipulable et compréhensible, et dont la simulation est couteuse (en temps). On retrouve dans cette catégorie la plupart des modèles dit "physiologiques", c'est-à-dire les modèles qui s'inspirent du fonctionnement biologique du système nerveux et tentent de le formaliser.  
		b. __Faire un compromis entre le réalisme du modèle et sa complexité__, ce qui donne des modèles moins précis et moins proche de la réalité, mais plus facilement manipulable, compréhensible et dont la simulation est rapide. On retrouve dans cette catégorie la plupart des modèles dit "phénoménologiques", c'est-à-dire les modèles qui, sans la contrainte de s'inspirer du biologique, tentent de reproduire le fonctionnement du système nerveux en terme de données quantitatives "de plus haut niveau", comme par exemple le potentiel de membrane d'un neurone. 
	
	Et de la deuxième proposition (2.) on peut dégager deux buts :  
		c. __Créer des machines douées de propriétés cognitives__ : On a donc ici une intentionalité qui appartent au domaine de l'intelligence artificielle ou de la cognition artificielle et une méthodologie de réalisation qui appartient au domaine de la modélisation du système nerveux. Il est logique de se rapprocher voire de se confondre avec ces champs recherche dès lors où notre intention, dans notre tache de modélisation, est celle de tenter de faire emerger des propriétés cognitives d'une machine. On peut ici préciser l'approche de l'IA-modèle-du-cerveau en la comparant à une approche plus classique en intelligence artificelle : celle des réseaux de neurones formels.  Le neurone : Modèle de neurone biologique / neurone formel  La connexion entre les neurones : modèle de synapses biologiques / connexions simple avec poids  La méthode d'apprentissage : Methode d'apprentissage s'inspirant de ce qu'on sait de l'apprentissage dans le système nerveux / surtout apprentissage supervisé,   Architecture : Inspirée de celle du cerveau / cherchant à maximiser l'efficacité du système, généralement choisie par un humain.  Il aurait été très interessant de débattre sur les questions : Est-ce que reproduire le fonctionnement du système nerveux humain est la voie la seule voie pour atteindre une machine avec une intelligence proche de l'humain (et sur quelle critère jugée de la réussite d'une telle entreprise ?) ? Est-ce la plus simple ? 
		d. __Mieux comprendre le fonctionnement du système nerveux__ : en effet posséder un modèle simulé par ordinateur du système nerveux ou d'une partie du système nerveux permettrait d'étudier l'impact de lésions dans un emplacement parfaitement controlées, d'avoir des données parfaitement "propres" et précises sur lesquelles travailler, de mettre en place beaucoup plus facilement des procèdures d'analyse en se servant des outils mathématiques et informatiques, etc ...
Par exemple, supposons que l'on dispose d'un modèle du système nerveux, que l'on subdivise ce modèle en plusieurs sous-parties, et que l'on souhaite savoir quel est l'ensemble minimal de sous-parties du modèle necessaire pour que le modèle possède une certaine capacité C. Supposons de plus (et c'est une supposition assez lourde) que l'on possède une mesure M capable de déterminer si un système possède la capacité C (M(C) vrai si le système possède C, faux sinon). Alos on peut envisager de mettre en place un algorithme du type :  
		```
		Pour toutes les sous-parties du système  
	Tenter d'enlever la sous-partie courante:  
	Si M(C) est vrai:  
		On enleve definitivement la sous-partie  
	sinon:  
		On réintègre la sous-partie dans le système
		```  
		Si les sous-parties en lesquelles on a subdivisé le système sont des zones spatiales, c'est-à-dire des ensembles de neurones (voire un neurone), alors cette algorithme revient, dans une approche plus classique, à faire des lésions successives de zones du cerveau. Si les sous-parties sont des propriétés des neurones ou des synapses, cela revient dans une approche classique, à bloquer successivement, à l'aide de composantes chimiques par exemple, certaines propriétés des neurones ou des synapses.  
Un modèle informatique du système nerveux permettrait de répondre à ce genre de question de manière certaine (à l'intérieur du modèle) et rapide. Il est également important d'insister sur la facilité d'acquistion de données aussi précises que l'on veut (dans la limite de la précision de l'ordinateur). 

	Maintenant, tout en gardant en tête toutes ces idées, il est temps pour moi de définir plus précisement l'objet de ce travail. Dans une première partie, je m'interesserai aux différents modèles de neurones existants et je créerai mon propre outil de définition et de simulation de ces modèles en Python 3+. Dans une deuxième partie, je me tournerai vers les modèles de synapses et je m'interesserai aux manières de créer et simuler des réseaux de neurones _in silico_, tout en créant mon propre outil de définition et de simulation de réseaux de spiking neurons en Python3+/Tensorflow. Dans une troisième partie, et à l'aide de l'outil créer dans la deuxième partie, je tenterai de reproduire les résultats de Herice et all 2016, c'est-à-dire construire un modèle des ganglions de la base sous forme de réseaux de spiking neurons, capable de prendre des décisions.
		
	(Apparté : Motivation personnelles) C'est donc avec ces idées (celles énoncées dans l'introduction pour ceux qui lisent dans le désordre) que j'aborde ce Travail Encadré de Recherche dans le cadre de ma troisème année de licence de Mathématiques et Informatique appliqués aux sciences humaines et sociales - Parcours Sciences Cognitive à l'université de Bordeaux. Ce TER a, je dois bien l'avouer, surtout une portée d'apprentissage pour moi, puisque j'espère qu'il me permettra d'acquérir les connaissances necessaire sur la modélisation du système nerveux pour ensuite poursuivre les buts énoncés dans l'introduction dans la suite de mes études, puis en recherche. Pour justifier la création de mes propres outils de définition et de simulation des modèles, je dirais que cela me permettra une compréhension profonde de la manière dont fonctionnent ces outils, notamment des méthodes de simulation numérique. De plus, il y a un manque d'outil de création de réseaux de spiking neurons en Python, et je pense que l'utilisation de tensorflow (bibliothèque de computation parallèle et optimisée) se revelera pertinente, et j'espère ainsi, au cours de se travail et après, créer une bibliothèque de création de modèles de réseaux de neurones biologiques facile d'utilisation et optimisée en terme de computation, du moins autant que cela est possible. Enfin l'étude et la reproduction des travaux de Héricé et all 2016 permettront, en plus de m'apporter de la connaissance et de la pratique sur la base d'une partie de la littérature récente sur le sujet, d'apporter une justification de l'utilité de mon outil de simulation de réseaux de neurones.
	

	(Apparté : les paradigmes d'étude psychologique de la cognition humaine) Parmis les grands paradigmes qui ont été explorés dans l'étude de l'esprit humain, on peut citer (manière non exhaustive) le behaviorisme, le cognitivisme et le connexionnisme. Le behaviorisme a tenté d'étudier l'humain en le réduisant à son comportement observable, c'est-à-dire ses interactions observables avec l'environnement. Cette approche est aujourd'hui considérée comme désuette, aussi, on ne s'attardera pas dessus. Le cognitivisme, paradigme fondateur des sciences cognitives, considère que le système nerveux humain peut (et doit) être étudié comme un système de traitement de l'information créant et manipulant des représentations symboliques et sémantiques du monde, ces représentations possédant des propriétés syntaxiques. Dans ce système cognitiviste, la pensée est comparée à une série d'application de règles (ou dit plus simplement, un calcul) sur les représentations. Le connexionnisme modélise les phénomènes mentaux comme des processus émergents de réseaux d'unités simples interconnectées. On oppose souvent cognitivisme et connexionnisme (ou pour être plus précis, computationnalisme et connexionnisme), en pensant que ce sont deux modes de pensées complétements opposés. Pourtant, les deux formulent la pensée comme un calcul sur des symboles. Les différences entre ces deux paradigmes se situent au niveau de la localisation des propriétés syntaxiques et sémantiques et dans la forme des règles de manipulation. Dans le cognitivisme, les propriétés syntaxiques et sémantiques sont attribuées aux représentations et les règles de manipulation sont algorithmiques et linéaires. Dans le paradigme connexioniste, les propriétés syntaxiques et les règles de calculs sont représentées par un réseau d'unités de bases interconnectées et dépendent entiéremment du fonctionnement de ces unités et de l'architecture du réseaux, et les règles de calculs peuvent être massivement parallèles ; tandis que les propriétés sémantiques sont attribuées au réseau en lui-même (en entier). On peut situer notre approche comme étant partie du courant connexionniste, meme si dans le cas de ce travail, ce n'est pas vraiment important.
	


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
	

1. Interets de réduire Hodgkin-Huxley (Quand faut-il privilegier la rapidite d'execution au réalisme biologique ?), réduction aux modèles de Fitzugh-Nagumo, modèle de Izhiekievich 17/02
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
from snn.single.usual_models import izhi_model as iz
import matplotlib.pyplot as plt

fig1 = iz.plot(1000, 1, keep=['I', 'u', 'v'])
fig2 = iz.plot(1000, 1, keep=['I', 'u', 'v'], subplotform='22')
fig3 = iz.plan_phase(('v', -80, -30, 5), ('u', -30, 20, 5),
                       rescale=True)
fig4 = iz.plan_phase(('v', -80, -30, 5), ('u', -30, 20, 5),
                          interactive=True, no_dynamics=True)
plt.show()
```

![alt Text](https://user-images.githubusercontent.com/27825602/36076476-e6d45882-0f5c-11e8-95d9-1e9a16462bbc.JPG)
![alt Text](https://user-images.githubusercontent.com/27825602/36076477-e7ec5936-0f5c-11e8-9830-e70f3b350b75.JPG)
![alt Text](https://user-images.githubusercontent.com/27825602/36076478-e993eede-0f5c-11e8-99e9-2dd3d4f44b07.JPG)
![alt Text](https://user-images.githubusercontent.com/27825602/36076474-e60e1bcc-0f5c-11e8-9dba-75ee33e47904.gif)


