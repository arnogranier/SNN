import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import os ; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np


def is_iterable(x):
    """Check if x is iterable"""
    try:
        iter(x)
        return True
    except:
        return False


def is_number(x):
    """Check if x is a Number"""
    return isinstance(x, float) or isinstance(x, int)


def treat_parameter(x, n=10, type_='classic', n1=None, n2=None, label=None):
    """Treat parameters with a name scope"""
    with tf.name_scope(label+'/'):
        if isinstance(x, (tf.Tensor, tf.Variable, tf.SparseTensor)):
            if type_ == 'connect_matrix' and x.shape == (n1, n2):
                return tf.cast(tf.transpose(x), tf.float32)
            return tf.cast(x, tf.float32)
        elif isinstance(x, np.ndarray):
            if type_ == 'connect_matrix' and x.shape == (n1, n2):
                return tf.Variable(x.T, dtype=tf.float32)
            return tf.Variable(x, dtype=tf.float32)
        elif is_number(x) and type_ == 'classic':
            return x * tf.ones((n, 1), dtype=tf.float32)
        elif is_number(x) and type_ == 'connect_matrix':
            return x * tf.ones((n2, n1), dtype=tf.float32)


def treat_callable(call, n=10):
    """Treat callable parameters"""
    if callable(call):
        return call
    elif is_number(call):
        return lambda t: call * np.ones((n, 1))
