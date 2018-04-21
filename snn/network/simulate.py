from .tools import *

def simulate(T, dt, graph, nuclei, data):
    """Simulate the tf model for T seconds with dt time step using parmeters 
       from build and where Iext is the function of time which give the 
       external current injected for each time 0<=t<T"""

    (vs, us, fireds, Is, vs_op, us_op, Is_op, fireds_op, external_inputs) = data

    with tf.Session(graph=graph) as sess:
        M = int(T / dt)
        sess.run(tf.global_variables_initializer())

        # Getting initialization values
        vs, us, Is, fireds = sess.run([vs, us, Is, fireds])
        vss = [[v, ] for v in vs]
        uss = [[u, ] for u in us]
        Iss = [[I, ] for I in Is]
        firedss = [{0: fired} for fired in fireds]
        # Initialization of the matrixes containg the states
        Iext = [N.Iext for N in nuclei]

        for m in range(M - 1):
            print('\r%s/%s' % (m, M-2), end='\r')

            # Running the simulation, inputing the external current
            # and getting current stat for the variables and neurons
            # that fired
            vs, us, fireds, Is = sess.run([vs_op, us_op, fireds_op, Is_op],
                                     feed_dict={external_inputs[i]:f(m * dt)
                                                for i, f in enumerate(Iext)})

            # Storing states
            for i, v in enumerate(vs) : vss[i].append(v)
            for i, u in enumerate(us) : uss[i].append(u)
            for i, I in enumerate(Is) : Iss[i].append(I)
            for i, fired in enumerate(fireds) :
                firedss[i][m * dt] = np.where(fired)[0]

    for i, nucleus in enumerate(nuclei):
        nucleus.historique['v'] = vss[i]
        nucleus.historique['u'] = uss[i]
        nucleus.historique['I'] = Iss[i]
        nucleus.historique['fired'] = firedss[i]

    return vss, uss, Iss, firedss