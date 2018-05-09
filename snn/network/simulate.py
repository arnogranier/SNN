from .tools import *


def simulate(T, dt, graph, nuclei, data):

    """Simulate the model in graph for T ms with a time step dt"""

    writer = tf.summary.FileWriter('./graph', graph)
    # Unpack data
    vs, us, fireds, Is, vs_op, us_op, Is_op, fireds_op, external_inputs = data

    # Actual tensorflow model simulation
    with tf.Session(graph=graph) as sess:
        M = int(T / dt)
        sess.run(tf.global_variables_initializer())

        # Get init variables state
        vs, us, Is, fireds = sess.run([vs, us, Is, fireds])
        vss = [[v, ] for v in vs]
        uss = [[u, ] for u in us]
        Iss = [[I, ] for I in Is]
        firedss = [{0: fired} for fired in fireds]
        Iext = [N.Iext for N in nuclei]

        # Looping through time
        for m in range(M - 1):
            print('\r%s/%s' % (m, M-2), end='\r')

            # Get data one time step further from the model
            feed = {external_inputs[i]: f(m * dt) for i, f in enumerate(Iext)}
            vs, us, fireds, Is = sess.run([vs_op, us_op, fireds_op, Is_op],
                                          feed_dict=feed)

            # Stocking data in lists
            for i, v in enumerate(vs):
                vss[i].append(v)
            for i, u in enumerate(us):
                uss[i].append(u)
            for i, I in enumerate(Is):
                Iss[i].append(I)
            for i, fired in enumerate(fireds):
                firedss[i][m * dt] = np.where(fired[-1,:])[0]

    # Stocking data in nuclei historic
    for i, nucleus in enumerate(nuclei):
        nucleus.historique['v'] = np.array(vss[i])
        nucleus.historique['u'] = np.array(uss[i])
        nucleus.historique['I'] = np.array(Iss[i])
        nucleus.historique['fired'] = firedss[i]

    return vss, uss, Iss, firedss
