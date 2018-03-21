import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt

T, dt = 100, 0.1
n = 10 ** 4
exi_inhi_rate = 8 / 10
v_init = -65

M = int(T / dt)

n_exi = round(n * exi_inhi_rate)
n_inhi = round(n * (1 - exi_inhi_rate))

v_shape = u_shape = I_shape = fired_shape = [n, 1]            
W_shape = [n, n]
exi_shape = [n_exi, 1] ; inhi_shape = [n_inhi, 1]

def model():

    graph = tf.Graph()
    with graph.as_default():

        W = np.random.rand(*W_shape)
        W[:n_exi, :] *= 0.5 ; W[n_exi:, :] *= -1
        W = tf.Variable(W, dtype=tf.float32)

        re = tf.random_uniform(exi_shape)
        ri = tf.random_uniform(inhi_shape)
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
        I = tf.Variable(tf.zeros(I_shape), dtype=tf.float32)

        new_v= tf.where(fired, c, v)
        new_u = tf.where(fired, tf.add(u, d), u)

        dv = tf.add(tf.subtract(tf.add(tf.multiply(tf.add(
                tf.multiply(0.04, new_v), 5.0), new_v), 140), new_u), I)
        new_v = tf.add(new_v, tf.multiply(dt, dv))
        du = tf.multiply(a, tf.subtract(tf.multiply(b, new_v), new_u))
        new_u = tf.add(new_u, tf.multiply(dt, du))

        fired_op = fired.assign(tf.greater_equal(new_v,
                                                 tf.ones(v_shape) * 30))
        new_v = tf.where(fired_op, tf.ones(v_shape) * 30, new_v)

        thalamic_inputs = tf.Variable(np.concatenate([np.random.randn(*exi_shape) * 5,
                                     np.random.randn(*inhi_shape) * 2]), dtype=tf.float32)
        internal_inputs = tf.expand_dims(tf.reduce_sum(
                                          tf.boolean_mask(W, tf.squeeze(fired)), 0), 1)
        I_op = I.assign(tf.add(thalamic_inputs, internal_inputs))
        v_op = v.assign(new_v)
        u_op = u.assign(new_u)
    return graph, v_op, u_op, fired_op, I_op

def simulate(graph, v_op, u_op, fired_op, I_op):
    fires = []
    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        for m in range(M):     
            _, _, fire, _ = sess.run(
                        [v_op, u_op, fired_op, I_op])
            fires.append(fire)
    return fires

import cProfile
graph, v_op, u_op, fired_op, I_op = model()
cProfile.run("simulate(graph, v_op, u_op, fired_op, I_op)")
# fires = simulate(graph, v_op, u_op, fired_op, I_op)
# plt.figure()
# xdata = np.array([]) ; ydata = np.array([])
# for t, f in enumerate(fires):
#     spikes_idx = np.where(f)[0]
#     xdata = np.concatenate([xdata, [t * dt] * len(spikes_idx)])
#     ydata = np.concatenate([ydata, spikes_idx])
# plt.plot(xdata, ydata, '|', color='black')
# plt.ylabel('Neuron indexes')
# plt.xlabel('time')
# plt.show()
