import numpy as np

def build(nb_of_neurons, proportion_ex_in, Iext, seed=123):

    np.random.seed(seed)

    nb_ex = round(nb_of_neurons * proportion_ex_in)
    nb_in = round(nb_of_neurons * (1 - proportion_ex_in))

    re = np.array(np.random.rand(nb_ex, 1), dtype=np.float32)
    ri = np.array(np.random.rand(nb_in, 1), dtype=np.float32)

    a = np.array(np.concatenate((0.02 * np.ones((nb_ex, 1)), 0.02 + 0.08 * ri)),dtype=np.float32)
    b = np.array(np.concatenate((0.2 * np.ones((nb_ex, 1)), 0.25 - 0.05 * ri)),dtype=np.float32)
    c = np.array(np.concatenate((- 65 + 15 * re ** 2, - 65 * np.ones((nb_in, 1)))),dtype=np.float32)
    d = np.array(np.concatenate((8 - 6 * re ** 2, 2 * np.ones((nb_in, 1)))),dtype=np.float32)

    W = np.random.rand(nb_of_neurons, nb_of_neurons) 
    W[:nb_ex, :] *= 0.5 ; W[nb_ex:, :] *= -1
    W = np.array(W, dtype=np.float32)

    v = - 65 * np.ones((nb_of_neurons, 1), dtype=np.float32)
    u = np.multiply(b,v)
    fired = np.zeros((nb_of_neurons), dtype=np.bool)
    I = Iext(0)

    return a, b, c, d, W, v, u, I, fired

def simulate(T, dt, a, b, c, d, W, v, u, I, fired, Iext, seed=123):
    np.random.seed(seed)
    
    M = int(T / dt)
    vs, us, Is, fireds = v.T, u.T, I.T, {0:fired}
    
    for m in range(M - 1):

        u = u + dt * (a * (b * v - u))
        v = v + dt * (((0.04 * v) + 5) * v + 140 - u + I)
        u[fired] = u[fired] + d[fired]
        v[fired] = c[fired]
        
        fired = np.where(v >= 30)[0]
        wheights_of_fired = np.sum(W[fired, :], axis=0, keepdims=True)
        I = Iext(m * dt) + wheights_of_fired.T
        Is = np.vstack((Is, I.T)) ; vs = np.vstack((vs, v.T))
        us = np.vstack((us, u.T))
        fireds[m * dt] = fired.copy()
    return (np.array(vs, dtype=np.float32), np.array(us, dtype=np.float32), 
            np.array(Is, dtype=np.float32), fireds)




