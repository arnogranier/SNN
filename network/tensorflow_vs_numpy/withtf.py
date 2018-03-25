import tensorflow as tf 
import numpy as np


def build(n, exi_inhi_rate, dt, Iext, seed=123):
    np.random.seed(seed)
    
    v_init = -65

    n_exi = round(n * exi_inhi_rate)
    n_inhi = round(n * (1 - exi_inhi_rate))

    v_shape = u_shape = I_shape = fired_shape = [n, 1]            
    W_shape = [n, n]
    exi_shape = [n_exi, 1] ; inhi_shape = [n_inhi, 1]

    graph = tf.Graph()
    with graph.as_default():
        
        re = tf.Variable(np.random.rand(*exi_shape), dtype=tf.float32)
        ri = tf.Variable(np.random.rand(*inhi_shape), dtype=tf.float32)
        
        W = np.random.rand(*W_shape)
        W[:n_exi, :] *= 0.5 ; W[n_exi:, :] *= -1
        W = tf.Variable(W.T, dtype=tf.float32)

        a = tf.Variable(tf.concat([0.02 * tf.ones(exi_shape),
                                   0.02 + 0.08 * ri], 0), dtype=tf.float32)
        b = tf.Variable(tf.concat([0.2 * tf.ones(exi_shape),
                                   0.25 - 0.05 * ri], 0), dtype=tf.float32)
        c = tf.Variable(tf.concat([-65 + 15 * re ** 2,
                                   -65 * tf.ones(inhi_shape)], 0), dtype=tf.float32)
        d = tf.Variable(tf.concat([8 - 6 * re ** 2,
                                   2 * tf.ones(inhi_shape)], 0), dtype=tf.float32)

        v = tf.Variable(tf.ones(shape=v_shape) * v_init,
                            dtype=tf.float32)
        u = tf.Variable(tf.multiply(b, v), dtype=tf.float32)
        fired = tf.Variable(tf.zeros(fired_shape, dtype=tf.bool))
        I = tf.Variable(Iext(0), dtype=tf.float32)

        new_v = tf.where(fired, c, v)
        new_u = tf.where(fired, tf.add(u, d), u)

        dv = tf.add(tf.subtract(tf.add(tf.multiply(tf.add(
                tf.multiply(0.04, new_v), 5.0), new_v), 140), new_u), I)
        new_v = tf.add(new_v, tf.multiply(dt, dv))
        du = tf.multiply(a, tf.subtract(tf.multiply(b, new_v), new_u))
        new_u = tf.add(new_u, tf.multiply(dt, du))

        fired_op = fired.assign(tf.greater_equal(new_v,
                                                 tf.ones(v_shape) * 30))

        external_input = tf.placeholder(tf.float32, shape=I_shape)
        I_op = I.assign(tf.add(external_input ,
                                tf.matmul(W, tf.cast(fired, tf.float32))))
        
        v_op = v.assign(new_v)
        u_op = u.assign(new_u)

        return graph, v, u, I, v_op, u_op, fired_op, I_op, external_input


def simulate(T, dt, graph, v, u, I, external_input, Iext, v_op, u_op, fired_op, I_op, seed=123):
    np.random.seed(seed)
    with tf.Session(graph=graph) as sess:
        M = int(T / dt)
        sess.run(tf.global_variables_initializer())
        v, u, I = sess.run([v, u, I])
        vs, us, fireds, Is = [v,], [u,], {0:np.array([])}, [I,] 
        for m in range(M-1):   
            v, u, fire, I = sess.run(
                        [v_op, u_op, fired_op, I_op], 
                        feed_dict={external_input:Iext(m * dt)})
            fireds[m * dt] = np.where(fire)[0]
            vs.append(v) ; us.append(u) ; Is.append(I)
    return (np.array(vs, dtype=np.float32), np.array(us, dtype=np.float32), 
            np.array(Is, dtype=np.float32), fireds)
