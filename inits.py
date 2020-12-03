import tensorflow as tf
import numpy as np
import math
import json


def uniform(shape, scale=0.05, name=None):
    initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones(shape, name=None):
    initial = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones_static(shape, name=None):
    initial = tf.ones(shape, dtype=tf.float32)
    return initial


def trunc_normal(shape, name=None, normalize=True):
    initial = tf.truncated_normal(shape, stddev=1.0 / math.sqrt(shape[0]))
    initial = tf.Variable(initial, name=name)
    if normalize:
        initial = tf.nn.l2_normalize(initial, 1)
    return initial


def one_trunc_normal(shape, name=None, normalize=True):
    initial = tf.truncated_normal(shape, stddev=1.0 / math.sqrt(shape[0]))
    initial = tf.Variable(initial, name=name)
    if normalize:
        initial = tf.nn.l2_normalize(initial, 1)
    return initial + 1.


def trunc_normal_2(shape, name=None, normalize=True):
    initial = tf.truncated_normal(shape, stddev=1.0 / math.sqrt(shape[0]))
    initial = tf.Variable(initial, name=name)
    if normalize:
        norm_1 = tf.nn.l2_normalize(initial[:, :int(shape[1]/2)], 1)
        norm_2 = tf.nn.l2_normalize(initial[:, int(shape[1]/2):], 1)
        initial = tf.concat([norm_1, norm_2],axis=1)
    return initial


def eyes(shape, name=None):
    initial = tf.eye(shape[0], num_columns=shape[1], dtype=tf.float32)
    return tf.Variable(initial, name=name)


def ones_vec(shape, name=None):
    initial = tf.ones(shape[0], dtype=tf.float32)
    return tf.diag(tf.Variable(initial, name=name))
