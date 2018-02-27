from brnn.single.usual_models import HH_model as hh
from brnn.single.usual_models import izhi_model as iz
import matplotlib.pyplot as plt
from brnn.single.tools import Function as F

# fig1 = iz.plot(1000, 1, keep=['I', 'u', 'v'])
# fig2 = iz.plot(1000, 1, keep=['I', 'u', 'v'], subplotform='22')
# fig3 = iz.plan_phase(('v', -80, -30, 5), ('u', -30, 20, 5),
#                        rescale=True)
# fig4 = iz.plan_phase(('v', -80, -30, 5), ('u', -30, 20, 5),
#                           interactive=True, no_dynamics=True)
# plt.show()

import random as rd 
import math

hh['Iapp'] = F('t', lambda t : 5*math.cos(t/10))
hh.method = 'rk4'
# fig1 = hh.plot(1000, 0.1, subplotform='32', keep=['alpha_n', 'beta_n', 'alpha_m', 'beta_m', 'alpha_h', 'beta_h'])
# fig2 = hh.plot(1000, 0.1, keep=['V', 'Iapp'])
# fig3 = hh.plot(1000, 0.1, keep=['m', 'n', 'h'])
history, _ = hh.simulation(1000, 0.01)
plt.plot(history['V'], history['n'])
plt.show()



































































