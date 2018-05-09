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
    fmax = int(max([delay for N in nuclei for (_, _, delay) in N.afference]) + 1)

    # Initialize the list of vectors representing the v variables
    with tf.name_scope('v'):
        vs = [tf.Variable(-65 * tf.ones((N.n, 1)), dtype=tf.float32)
          for N in nuclei]

    # Initialize the list of vectors representing the u variables
    with tf.name_scope('u'):
        us = [tf.Variable(tf.multiply(N.b, v), dtype=tf.float32)
          for (N, v) in zip(nuclei, vs)]

    # Initialize the list of boolean vectors representing when neurons fired
    with tf.name_scope('fireds'):    
        fireds = [tf.Variable(tf.cast(tf.zeros((fmax, N.n)), tf.bool))
              for N in nuclei]

    # Initialize the list of vectors representing the input I
    with tf.name_scope('I'):    
        Is = [tf.Variable(tf.zeros((N.n, 1)), dtype=tf.float32) for N in nuclei]

    # Initialize the list of vectors to stock I
    # It's necessary to stock I when we compute dvs, because we update I in parallel
    # of computing dvs, and I is in dvs equation.
    with tf.name_scope('I_stock'):
        I_stock = [tf.Variable(tf.zeros((N.n, 1)), dtype=tf.float32) for N in nuclei]
        I_stock_op = [tf.assign(stock, I) for (stock, I) in zip(I_stock, Is)]

    # Initialize the list of placeholders for external input
    with tf.name_scope('external_inputs'):
        external_inputs = [tf.placeholder(tf.float32, shape=(N.n, 1))
                       for N in nuclei]

    # Reset rules
    with tf.name_scope('last_fired'):    
        last_fired = [tf.gather(fired, fmax - 1) for fired in fireds]
    with tf.name_scope('v_reseted'):
        new_vs = [tf.where(fired, N.c, v)
              for (v, fired, N) in zip(vs, last_fired, nuclei)]
    with tf.name_scope('u_reseted'):
        new_us = [tf.where(fired, tf.add(u, N.d), u)
              for (u, fired, N) in zip(us, last_fired, nuclei)]

    # Dynamical system
    cst004 = tf.constant(0.04, dtype=tf.float32)
    cst5 = tf.constant(5, dtype=tf.float32)
    cst140 = tf.constant(140, dtype=tf.float32)
    with tf.name_scope('dv'):
        dvs = [tf.add(tf.subtract(tf.add(tf.multiply(tf.add(
            tf.multiply(cst004, new_v), cst5), new_v), cst140), new_u), I)
           for (new_v, new_u, I) in zip(new_vs, new_us, I_stock_op)]
    with tf.name_scope('after_euler_step_v'):    
        new_vs = [np.add(new_v, np.multiply(dv, dt))
              for (new_v, dv) in zip(new_vs, dvs)]
    with tf.name_scope('du'):
        dus = [tf.multiply(N.a, tf.subtract(tf.multiply(N.b, new_v), new_u))
           for (new_v, new_u, N) in zip(new_vs, new_us, nuclei)]
    with tf.name_scope('after_euler_step_u'):
        new_us = [np.add(new_u, np.multiply(du, dt))
              for (new_u, du) in zip(new_us, dus)]

    # Check neurons that fired and update fired
    with tf.name_scope('v30'):
        v30s = [tf.Variable(30 * tf.ones((N.n, 1)), dtype=tf.float32)
            for N in nuclei]
    with tf.name_scope('all_except_last'):
        all_except_last = tf.constant(np.arange(1, fmax), dtype=tf.int32)
    with tf.name_scope('fireds_op'):
        fireds_op = [fired.assign(tf.concat([tf.gather(fired, all_except_last),
                        tf.transpose(tf.greater_equal(new_v, v30))], 0))
                 for (fired, new_v, v30) in zip(fireds, new_vs, v30s)]

    # keep v of neurons that fired at 30mV
    with tf.name_scope('v_kept_at_30'):
        vs_reseted = [tf.where(tf.gather(fired_op, fmax - 1), v30, new_v)
                  for (fired_op, v30, new_v) in zip(fireds_op, v30s, new_vs)]

    # Update v and u
    with tf.name_scope('v_op'):
        vs_op = [v.assign(v_reseted) for (v, v_reseted) in zip(vs, vs_reseted)]
    with tf.name_scope('u_op'):    
        us_op = [u.assign(new_u) for (u, new_u) in zip(us, new_us)]

    # afference from other nuclei
    with tf.name_scope('from_other_nuclei'):
        from_other_nuclei = [tf.add_n(
       [tf.zeros(vs[ni].shape)] +
       [tf.matmul(P, tf.cast(tf.expand_dims(tf.gather(fireds[nuclei.index(M)],
                                                       fmax - 1 - delay), 1),
                              tf.float32))
        for (M, P, delay) in N.afference])
                         for ni, N in enumerate(nuclei)]

    # Compute input (external, internal and from other nuclei) and update I
    with tf.name_scope('I_op'):
        Is_op = [I.assign(tf.add_n(
       [external_input, 
        tf.matmul(N.W, tf.cast(tf.expand_dims(tf.gather(fired, fmax - 1), 1),
                               tf.float32)),
        from_other]))
        for (N, external_input, fired, from_other, I)
        in zip(nuclei, external_inputs, fireds, from_other_nuclei, Is)]

    return [vs, us, fireds, Is, vs_op, us_op,
            Is_op, fireds_op, external_inputs]
