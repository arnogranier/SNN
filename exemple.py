import math
import random as rd 
from snn.single.tools import Function as F
from snn.single.usual_models import HH_model as hh
import matplotlib.pyplot as plt

# Define Input current
#hh['Iapp'] = F('t', lambda t : 5 if 200<t<400 else 10 if 600<t<800 else 0)
hh['Iapp'] = 10
#hh['Iapp'] = F('t', lambda t : 2*math.cos(t / 10) + 2.5*rd.expovariate(0.5))
# Runge-Kutta method for numerical simulation
hh.method = 'rk4'

# Since we are plotting multiple things, it's better to simulate the
# model only one time, and then feed the results to the plot method
T, dt = 12, 0.01
history, _ = hh.simulation(T, dt)

# Plot the input current and the membrane potential evolution throught time
fig1 = hh.plot(T, dt, keep=['V', 'Iapp'], history=history)

# Plot the evolution of m, n, h through time
fig2 = hh.plot(T, dt, keep=['m', 'n', 'h'], #subplotform='31', 
			   history=history)

# Print m, n, h evolution when the membrane potentiel changes
fig3 = plt.figure() ; plt.ylabel('n'), plt.xlabel('V')
plt.plot(history['V'], history['n'])
fig4 = plt.figure() ; plt.ylabel('m'), plt.xlabel('V')
plt.plot(history['V'], history['m']) 
fig5 = plt.figure() ; plt.ylabel('h'), plt.xlabel('V')
plt.plot(history['V'], history['h']) 

test=plt.figure()

plt.plot(history['V'], 1/(history['alpha_n']+history['beta_n']))
plt.plot(history['V'], 1/(history['alpha_m']+history['beta_m']))
plt.plot(history['V'], 1/(history['alpha_h']+history['beta_h']))

test2 = plt.figure()
#print([round(x, 1) for x in list(history['beta_h'])])
plt.plot(history['V'], history['alpha_n']/(history['alpha_n']+history['beta_n']))
plt.plot(history['V'], history['alpha_m']/(history['alpha_m']+history['beta_m']))
plt.plot(history['V'], history['alpha_h']/(history['alpha_h']+history['beta_h']))
test.show()

test2.show()




"""from snn.single.usual_models import izhi_model as iz
import matplotlib.pyplot as plt

fig1 = iz.plot(1000, 1, keep=['I', 'u', 'v'])
fig2 = iz.plot(1000, 1, keep=['I', 'u', 'v'], subplotform='22')
fig3 = iz.plan_phase(('v', -80, -30, 5), ('u', -30, 20, 5),
                       rescale=True)
fig4 = iz.plan_phase(('v', -80, -30, 5), ('u', -30, 20, 5),
                          interactive=True, no_dynamics=True)
plt.show()"""




























































