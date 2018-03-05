import numpy as np 
import matplotlib.pyplot as plt

T = 10 ; dt = 0.1 ; methode = 3

M = int(T/dt)
x = np.linspace(0, T, M)
exact = -2*np.exp(-5*x)+3*np.exp(-4*x)
approche = [1, ]
for i in range(M-1):
    un = approche[-1]
    if methode == 1: approche.append((un+2*dt*np.exp(-5*i*dt))/(1+4*dt) )
    elif methode == 2 : approche.append(un + dt*(-4*un+2*np.exp(-5*i*dt)))
    elif methode == 3 : approche.append((un + (dt/2)*(-4*un+2*np.exp(-5*i*dt)+2*np.exp(-5*(i+1)*dt)))/(1+2*dt))
print('Erreur : %s' % sum(abs(exact-approche)))
plt.plot(x, exact, '-b')
plt.plot(x, approche, '--r')
plt.show()

