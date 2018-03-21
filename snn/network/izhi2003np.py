import numpy as np

def simulate(nb_of_neurons, proportion_ex_in, T, dt):
    M = int(T / dt)

    # Nombre de neurones excitateurs et inhibiteurs en fonction 
    # du nombre total de neurones et de la proportion excitateurs/inhibiteurs
    nb_ex = round(nb_of_neurons * proportion_ex_in)
    nb_in = round(nb_of_neurons * (1 - proportion_ex_in)) 

    # Valeurs aleatoires (pour la diversite des types de neurones)
    re = np.random.rand(nb_ex, 1) ; ri = np.random.rand(nb_in, 1)

    # Vecteurs des parametres du modele
    a = np.concatenate((0.02 * np.ones((nb_ex, 1)), 0.02 + 0.08 * ri))
    b = np.concatenate((0.2 * np.ones((nb_ex, 1)), 0.25 - 0.05 * ri))
    c = np.concatenate((- 65 + 15 * re ** 2, - 65 * np.ones((nb_in, 1))))
    d = np.concatenate((8 - 6 * re ** 2, 2 * np.ones((nb_in, 1))))

    # Matrice des poids
    S = np.random.rand(nb_of_neurons, nb_of_neurons) 
    S[:nb_ex, :] *= 0.5 ; S[nb_ex:, :] *= -1

    # Initialisation des vecteurs des valeurs des variables du modele
    v = - 65 * np.ones((nb_of_neurons, 1))
    u = np.multiply(b,v)

    x_coord, y_coord = np.array([]), np.array([])
    for t in range(M):
        #print(t)
        # Input aleatoire
        I = np.concatenate((5 * np.random.randn(nb_ex, 1),
                      2 * np.random.randn(nb_in, 1)))

        # Indices des neurones qui ont produit un spike
        fired = np.where(v >= 30)[0]

        # Ajout des coordonnees des points a afficher
        x_coord = np.concatenate((x_coord, t + 0 * fired))
        y_coord = np.concatenate((y_coord, fired))

        # Regle speciale pour les neurones ayant decharge
        v[fired] = c[fired]
        u[fired] = u[fired] + d[fired]

        # Apport des neurones a l'input de chaque autre neurone
        wheights_of_fired = np.sum(S[fired, :], axis=0, keepdims=True)
        I = I + wheights_of_fired.T

        # Calcul des nouvelles valeurs des variables du modele par
        # la methode d'Euler
        v = v + dt * (0.04 * v ** 2 + 5 * v + 140 - u + I)
        u = u + dt * (a * (b * v - u))

    return x_coord, y_coord

# Parametres
nb_of_neurons, proportion_ex_in, T, dt = 10 ** 4, 8 / 10, 100, 0.1
import cProfile
cProfile.run('simulate(nb_of_neurons, proportion_ex_in, T, dt)')

# import matplotlib.pyplot as plt
# fig = plt.figure()

#   # Affichage de type raster plot
# plt.plot(x_coord, y_coord, '|', color='black' )

#   # Titre, Labels de axes
# plt.title('Raster plot of %s neurons network during %s ms' % 
#                           (nb_of_neurons, T))
# plt.xlabel('t(ms)') ; plt.ylabel('neuron N')
    
# fig.show()
