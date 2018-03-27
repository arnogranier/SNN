import tensorflow as tf
import numpy as np


def build(n, exi_inhi_rate, dt, Iext, seed=123):
    """Crating tf graph for simple network used in test np vs tf"""

    np.random.seed(seed)

    n_exi = round(n * exi_inhi_rate)
    n_inhi = round(n * (1 - exi_inhi_rate))

    graph = tf.Graph()
    with graph.as_default():

        # Parameters generation as in (Izhikievich, 2003) in order to have
        # a variety of neural behavior
        re = tf.Variable(np.random.rand(n_exi, 1), dtype=tf.float32)
        ri = tf.Variable(np.random.rand(n_inhi, 1), dtype=tf.float32)
        a = tf.Variable(tf.concat([0.02 * tf.ones((n_exi, 1)),
                                   0.02 + 0.08 * ri], 0), dtype=tf.float32)
        b = tf.Variable(tf.concat([0.2 * tf.ones((n_exi, 1)),
                                   0.25 - 0.05 * ri], 0), dtype=tf.float32)
        c = tf.Variable(tf.concat([-65 + 15 * re ** 2,
                                   -65 * tf.ones((n_inhi, 1))], 0),
                        dtype=tf.float32)
        d = tf.Variable(tf.concat([8 - 6 * re ** 2,
                                   2 * tf.ones((n_inhi, 1))], 0),
                        dtype=tf.float32)

        # Weight matrix generation W =(wij) wich give the weight of the 
        # connexion between the neuron j and the neuron i
        # Excitator neurons have positive weights between 0 and 0.5
        # Inhibitor neurons have negative weights between -1 and 0
        W = np.random.rand(n, n)
        W[:n_exi, :] *= 0.5
        W[n_exi:, :] *= -1
        W = tf.Variable(W.T, dtype=tf.float32)

        # Initialisation state of the variables
        v = tf.Variable(-65 * tf.ones((n, 1)), dtype=tf.float32)
        u = tf.Variable(tf.multiply(b, v), dtype=tf.float32)
        fired = tf.Variable(tf.zeros((n, 1), dtype=tf.bool))
        I = tf.Variable(Iext(0), dtype=tf.float32)

        # Reset rule
        new_v = tf.where(fired, c, v)
        new_u = tf.where(fired, tf.add(u, d), u)

        # The dynamical system
        dv = tf.add(tf.subtract(tf.add(tf.multiply(tf.add(
                tf.multiply(0.04, new_v), 5.0), new_v), 140), new_u), I)
        new_v = np.add(new_v, np.multiply(dv, dt))
        du = tf.multiply(a, tf.subtract(tf.multiply(b, new_v), new_u))
        new_u = np.add(new_u, np.multiply(du, dt))

        # Get a boolean vector of neurons that fired
        v30 = tf.Variable(30 * tf.ones((n, 1)), dtype=tf.float32)
        fired_op = fired.assign(tf.greater_equal(new_v, v30))

        # Keep all neurons that fired at 30mV
        v_reseted = tf.where(fired_op, v30, new_v)
        v_op = v.assign(v_reseted)
        u_op = u.assign(new_u)

        # Update the input, according to external input and to
        # internal influence of neurons that fired, 
        # wich is computed by doing W.f with f=(fi) the n-tuple
        # with fi = 1 if neuron i fired else fi = 0
        external_input = tf.placeholder(tf.float32, shape=(n, 1))
        I_op = I.assign(tf.add(external_input,
                               tf.matmul(W, tf.cast(fired, tf.float32))))

        return graph, v, u, I, v_op, u_op, fired_op, I_op, external_input


def simulate(T, dt, graph, v, u, I, external_input, Iext,
             v_op, u_op, fired_op, I_op, seed=123):
    """Simulate the tf model for T seconds with dt time step using parmeters 
       from build and where Iext is the function of time which give the 
       external current injected for each time 0<=t<T"""

    np.random.seed(seed)

    with tf.Session(graph=graph) as sess:
        M = int(T / dt)
        sess.run(tf.global_variables_initializer())

        # Getting initialization values
        v, u, I = sess.run([v, u, I])

        # Initialization of the matrixes containg the states
        vs, us, fireds, Is = [v, ], [u, ], {0: np.array([])}, [I, ]

        for m in range(M - 1):

            # Running the simulation, inputing the external current
            # and getting current stat for the variables and neurons
            # that fired
            v, u, fire, I = sess.run([v_op, u_op, fired_op, I_op],
                                     feed_dict={external_input: Iext(m * dt)})

            # Storing states
            Is.append(I)
            vs.append(v)
            us.append(u)
            fireds[m * dt] = np.where(fire)[0]

    return (np.array(vs, dtype=np.float32), np.array(us, dtype=np.float32),
            np.array(Is, dtype=np.float32), fireds)
