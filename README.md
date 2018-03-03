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
voir tertext/ter.pdf

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


