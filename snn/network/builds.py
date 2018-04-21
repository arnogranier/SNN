from .tools import *

def build_izhi(dt, nuclei):
    # Initialisation state of the variables
    vs = [tf.Variable(-65 * tf.ones((N.n, 1)), dtype=tf.float32) for N in nuclei]
    us = [tf.Variable(tf.multiply(N.b, v), dtype=tf.float32) for (N,v) in zip(nuclei, vs)]
    fireds = [tf.Variable(tf.zeros((N.n, 1), dtype=tf.bool)) for N in nuclei]
    Is = [tf.Variable(tf.zeros((N.n, 1)), dtype=tf.float32) for N in nuclei]
    external_inputs = [tf.placeholder(tf.float32, shape=(N.n, 1)) for N in nuclei]
    # Reset rule
    new_vs = [tf.where(fired, N.c, v) for (v, fired, N) in zip(vs, fireds, nuclei)]
    new_us = [tf.where(fired, tf.add(u, N.d), u) 
              for (u, fired, N) in zip(us, fireds, nuclei)]

    # The dynamical system
    dvs = [tf.add(tf.subtract(tf.add(tf.multiply(tf.add(
            tf.multiply(0.04, new_v), 5.0), new_v), 140), new_u), I) 
            for (new_v, new_u, I) in zip(new_vs, new_us, Is)]
    new_vs = [np.add(new_v, np.multiply(dv, dt)) for (new_v, dv) in zip(new_vs, dvs)]
    dus = [tf.multiply(N.a, tf.subtract(tf.multiply(N.b, new_v), new_u)) 
            for (new_v, new_u, N) in zip(new_vs, new_us, nuclei)]
    new_us = [np.add(new_u, np.multiply(du, dt)) for (new_u, du) in zip(new_us, dus)]

    # Get a boolean vector of neurons that fired
    v30s = [tf.Variable(30 * tf.ones((N.n, 1)), dtype=tf.float32) for N in nuclei]
    fireds_op = [fired.assign(tf.greater_equal(new_v, v30)) 
                 for (fired, new_v, v30) in zip(fireds, new_vs, v30s)]

    # Keep all neurons that fired at 30mV
    vs_reseted = [tf.where(fired_op, v30, new_v) 
                 for (fired_op, v30, new_v) in zip(fireds_op, v30s, new_vs)]
    vs_op = [v.assign(v_reseted) for (v, v_reseted) in zip(vs, vs_reseted)]
    us_op = [u.assign(new_u) for (u, new_u) in zip(us, new_us)]

    # Update the input, according to external input and to
    # internal influence of neurons that fired, 
    # wich is computed by doing W.f with f=(fi) the n-tuple
    # with fi = 1 if neuron i fired else fi = 0
    afference_idxs = [[(nuclei.index(M), P) for (M, P) in N.afference] for N in nuclei]
    from_other_nuclei = [tf.add_n([tf.zeros(vs[ni].shape)]+[tf.matmul(P, tf.cast(fireds[idx], tf.float32)) for (idx, P) in data]) 
                         for ni, data in enumerate(afference_idxs)]
    
    Is_op = [I.assign(tf.add_n([external_input,
                           tf.matmul(N.W, tf.cast(fired, tf.float32)), aff]))
                        for (N, external_input, fired, aff, I) in zip(nuclei, external_inputs, fireds, from_other_nuclei, Is)]

    return [vs, us, fireds, Is, vs_op, us_op, Is_op, fireds_op, external_inputs]