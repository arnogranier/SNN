from .tools import *


def build_izhi(dt, nuclei):

    """build a izhi model with a list of nucleus

        dv/dt = 0.04 * v ^ 2 + 5 * v + 140 - u + I
        du/dt = a * (b * v - u)

        if v >= 30 :
        v <- c
        u <- u + d
    """

    # Compute the needed length of stored fireds
    fmax = max([delay for N in nuclei for (_, _, delay) in N.afference]) + 1

    # Initialize the list of vectors representing the v variables
    vs = [tf.Variable(-65 * tf.ones((N.n, 1)), dtype=tf.float32)
          for N in nuclei]

    # Initialize the list of vectors representing the u variables
    us = [tf.Variable(tf.multiply(N.b, v), dtype=tf.float32)
          for (N, v) in zip(nuclei, vs)]

    # Initialize the list of boolean vectors representing when neurons fired
    fireds = [tf.Variable(tf.cast(tf.zeros((fmax, N.n)), tf.bool))
              for N in nuclei]

    # Initialize the list of vectors representing the input I
    Is = [tf.Variable(tf.zeros((N.n, 1)), dtype=tf.float32) for N in nuclei]

    # Initialize the list of placeholders for external input
    external_inputs = [tf.placeholder(tf.float32, shape=(N.n, 1))
                       for N in nuclei]

    # Reset rules
    new_vs = [tf.where(tf.gather(fired, fmax - 1), N.c, v)
              for (v, fired, N) in zip(vs, fireds, nuclei)]
    new_us = [tf.where(tf.gather(fired, fmax - 1), tf.add(u, N.d), u)
              for (u, fired, N) in zip(us, fireds, nuclei)]

    # Dynamical system
    dvs = [tf.add(tf.subtract(tf.add(tf.multiply(tf.add(
            tf.multiply(0.04, new_v), 5.0), new_v), 140), new_u), I)
           for (new_v, new_u, I) in zip(new_vs, new_us, Is)]
    new_vs = [np.add(new_v, np.multiply(dv, dt))
              for (new_v, dv) in zip(new_vs, dvs)]
    dus = [tf.multiply(N.a, tf.subtract(tf.multiply(N.b, new_v), new_u))
           for (new_v, new_u, N) in zip(new_vs, new_us, nuclei)]
    new_us = [np.add(new_u, np.multiply(du, dt))
              for (new_u, du) in zip(new_us, dus)]

    # Check neurons that fired and update fired
    v30s = [tf.Variable(30 * tf.ones((N.n, 1)), dtype=tf.float32)
            for N in nuclei]
    fireds_op = [fired.assign(tf.concat([tf.gather(fired, tf.range(1, fmax)),
                        tf.transpose(tf.greater_equal(new_v, v30))], 0))
                 for (fired, new_v, v30) in zip(fireds, new_vs, v30s)]

    # keep v of neurons that fired at 30mV
    vs_reseted = [tf.where(tf.gather(fired_op, fmax - 1), v30, new_v)
                  for (fired_op, v30, new_v) in zip(fireds_op, v30s, new_vs)]

    # Update v and u
    vs_op = [v.assign(v_reseted) for (v, v_reseted) in zip(vs, vs_reseted)]
    us_op = [u.assign(new_u) for (u, new_u) in zip(us, new_us)]

    # afference from other nuclei
    from_other_nuclei = [tf.add_n(
       [tf.zeros(vs[ni].shape)] +
       [tf.matmul(P, tf.cast(tf.expand_dims(tf.gather(fireds[nuclei.index(M)],
                                                       fmax - 1 - delay), 1),
                              tf.float32))
        for (M, P, delay) in N.afference])
                         for ni, N in enumerate(nuclei)]

    # Compute input (external, internal and from other nuclei) and update I
    Is_op = [I.assign(tf.add_n(
       [external_input, 
        tf.matmul(N.W, tf.cast(tf.expand_dims(tf.gather(fired, fmax - 1), 1),
                               tf.float32)),
        from_other]))
        for (N, external_input, fired, from_other, I)
        in zip(nuclei, external_inputs, fireds, from_other_nuclei, Is)]

    return [vs, us, fireds, Is, vs_op, us_op,
            Is_op, fireds_op, external_inputs]
