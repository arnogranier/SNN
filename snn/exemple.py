import matplotlib.pyplot as plt
from snn.single.usual_models import HH_model as hh

# Define Input current
hh['Iapp'] = 5
# Runge-Kutta method for numerical simulation
hh.method = 'rk4'

# Since we are plotting multiple things, it's better to simulate the
# model only one time, and then feed the results to the plot method
T, dt = 100, 0.01
history, _ = hh.simulation(T, dt)

# Plot the input current and the membrane 
# potential evolution throught time
hh.plot(T, dt, keep=['V', 'Iapp'], history=history)

# Print m, n, h evolution when the membrane potentiel changes
for var in ['n', 'm', 'h']:
	plt.figure() ; plt.ylabel(var) ; plt.xlabel('V')
	plt.plot(history['V'], history[var])

plt.show()































