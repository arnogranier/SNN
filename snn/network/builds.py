from .tools import *


def build_izhi(dt, nuclei, synapse_type='simple'):
    """
    Build a izhi model with a list of nucleus

        dv/dt = 0.04 * v ^ 2 + 5 * v + 140 - u + I
        du/dt = a * (b * v - u)

        if v >= 30 :
        v <- c
        u <- u + d

    Parameters
    ----------
    dt : float
        time step in seconds
    nuclei : list of Izhi_Nucleus
        List containing all the necessary information to build the network
    synapse_type : str
        Synapse type, either 'simple' for simple time decaying synapse model,
        or 'voltage_jump' for a simple instant voltage jump model

    Returns
    -------
    list of tensorflow operations
        The list of tensorflow operations to be executed

    """


    # Compute the needed length of stored fireds
    fmax = int((1 / dt) * max([howfar+delay for N in nuclei
                               for (_, _, delay, _, howfar) in N.afference]))+1

    # Initialize the list of vectors representing the v variables
    with tf.name_scope('v'):
        vs = [tf.Variable(N.c * tf.ones((N.n, 1)), dtype=tf.float32)
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
        Is = [tf.Variable(tf.zeros((N.n, 1)), dtype=tf.float32)
              for N in nuclei]

    # Initialize the list of vectors to stock I
    # It's necessary to stock I when we compute dvs, because we update I
    # in parallel of computing dvs, and I is in dvs equation.
    with tf.name_scope('I_stock'):
        I_stock = [tf.Variable(tf.zeros((N.n, 1)), dtype=tf.float32)
                   for N in nuclei]
        I_stock_op = [stock.assign(I) for (stock, I) in zip(I_stock, Is)]

    # Initialize the list of placeholders for external input
    with tf.name_scope('external_inputs'):
        external_inputs = [tf.placeholder(shape=(N.n, 1), dtype=tf.float32)
                           for N in nuclei]

    # Reset rules
    with tf.name_scope('last_fired'):
        last_fired = [tf.expand_dims(tf.gather(fired, fmax - 1), 1)
                      for fired in fireds]
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
        # dv/dt = 0.04 * v ^ 2 + 5 * v + 140 - u + I
        dvs = [tf.add(tf.subtract(tf.add(tf.multiply(tf.add(
                 tf.multiply(cst004, new_v), cst5), new_v), cst140), new_u), I)
               for (new_v, new_u, I) in zip(new_vs, new_us, I_stock_op)]
    with tf.name_scope('after_euler_step_v'):
        # v = v + dt*dv/dt
        new_vs = [tf.add(new_v, tf.multiply(dv, dt))
                  for (new_v, dv) in zip(new_vs, dvs)]
    with tf.name_scope('du'):
        # du/dt = a * (b * v - u)
        dus = [tf.multiply(N.a, tf.subtract(tf.multiply(N.b, new_v), new_u))
               for (new_v, new_u, N) in zip(new_vs, new_us, nuclei)]
    with tf.name_scope('after_euler_step_u'):
        # u = u + dt*du/dt
        new_us = [tf.add(new_u, tf.multiply(du, dt))
                  for (new_u, du) in zip(new_us, dus)]
        us_op = [u.assign(new_u) for (u, new_u) in zip(us, new_us)]

    # Check neurons that fired and update fired
    with tf.name_scope('v30'):
        v30s = [tf.constant(30 * np.ones((N.n, 1)), dtype=tf.float32)
                for N in nuclei]

    # Compute new fireds event wrt the variable v
    # Slide window of the fireds events kept in memory (to compute inputs
    # with the synapse model
    with tf.name_scope('new_fired'):
        new_fireds = [tf.transpose(tf.greater_equal(new_v, v30))
                     for (new_v, v30) in zip(new_vs, v30s)]
        all_except_last = tf.constant(np.arange(1, fmax), dtype=tf.int32)
        fireds_op = list()
        for (fired, new_fired) in zip(fireds, new_fireds):
            prev_fired = tf.gather(fired, all_except_last)
            new_slided_window_fired = tf.concat([prev_fired, new_fired], 0)
            fireds_op.append(fired.assign(new_slided_window_fired))


    # keep v of neurons that fired at 30mV
    with tf.name_scope('v_kept_at_30'):
        vs_reseted = [tf.where(tf.transpose(new_fired), v30, new_v)
                      for (new_fired, v30, new_v) in zip(new_fireds, v30s, new_vs)]
        vs_op = [v.assign(v_reseted) for (v, v_reseted) in zip(vs, vs_reseted)]

    # afference from other nuclei
    with tf.name_scope('internal_external_inputs'):
        if synapse_type == 'simple':
            decays = list()
            internal_inputs = list()
            for ni, N in enumerate(nuclei):
                decays_N = list()
                decays.append(decays_N)
                for (M, P, delay, decay, howfar) in N.afference:
                    times = np.linspace(int((1/dt)*howfar)-1, 0, int((1/dt)*howfar))
                    decay_values = np.expand_dims(decay(times),1)
                    decay = tf.convert_to_tensor(decay_values, dtype=tf.float32)
                    decays_N.append(decay)
            for (M, P, delay, _, howfar), decay in zip(N.afference, decays[ni]):
                stock = list()
                f_idxs = tf.range(fmax-int((1/dt) * delay) - int((1/dt)*howfar),
                                  fmax - int((1 / dt) * delay))
                ensure_shape_if_no_inputs = tf.zeros(vs[ni].shape)
                fired_idxs = tf.gather(fireds[nuclei.index(M)], f_idxs)
                float_idxs = tf.transpose(tf.cast(fired_idxs, tf.float32))
                stock.append(tf.matmul(tf.matmul(P, float_idxs), decay)
                             + ensure_shape_if_no_inputs)
            internal_inputs.append(tf.add_n(stock))
        elif synapse_type == 'voltage_jump':
            internal_inputs = list()
            for ni, N in enumerate(nuclei):
                stock = list()
                for (M, P, delay, decay, howfar) in N.afference:
                    ensure_shape_if_no_inputs = tf.zeros(vs[ni].shape)
                    fired_idxs = tf.gather(fireds[nuclei.index(M)],
                                           fmax-1-int((1 / dt)*delay))
                    float_idxs = tf.cast(tf.expand_dims(fired_idxs, 1),
                                         tf.float32)
                    stock.append(tf.matmul(P, float_idxs))
            internal_inputs.append(tf.add_n(stock))

        # Compute input (external, internal and from other nuclei) and update I
        Is_op = [I.assign(tf.add_n([external_input, internal_input]))
                 for (external_input, internal_input, I)
                 in zip(external_inputs, internal_inputs, Is)]

    return [vs, us, fireds, Is, vs_op, us_op,
            Is_op, fireds_op, external_inputs]
