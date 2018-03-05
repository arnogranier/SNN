from snn.single.usual_models import izhi_model as iz
import matplotlib.pyplot as plt

fig1 = iz.plot(1000, 1, keep=['I', 'u', 'v'])
fig2 = iz.plot(1000, 1, keep=['I', 'u', 'v'], subplotform='22')
fig3 = iz.plan_phase(('v', -80, -30, 5), ('u', -30, 20, 5),
                       rescale=True)
fig4 = iz.plan_phase(('v', -80, -30, 5), ('u', -30, 20, 5),
                          interactive=True, no_dynamics=True)
plt.show()




























































