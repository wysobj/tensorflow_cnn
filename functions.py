import tensorflow as tf
import numpy as np

def cosin(v1, v2):
    mul = np.dot(v1, v2)
    mod1 = np.sqrt(np.sum(np.power(v1, 2)))
    mod2 = np.sqrt(np.sum(np.power(v2, 2)))
    result = mul / (mod1 * mod2);
    return result

def euclid_distance(v1, v2):
    return np.sqrt(np.sum(np.power((v1 - v2), 2)))

def manhattan_distance(v1, v2):
    return np.sum(np.abs(v1 - v2))

def dim(x):
    shape = x.get_shape().as_list()
    return np.prod(shape[1:])

def weight_init(shape, w_init=None):
    if len(shape) == 2:
        in_dim = shape[0]
    if len(shape) == 4:
        in_dim = np.prod(shape[:-1])
    W = tf.Variable(tf.truncated_normal(shape, mean=0., stddev=0.01, dtype=tf.float32))
    decay_term = tf.nn.l2_loss(W)
    return W, decay_term

def dense_layer(x, hdim, w_init=None, b_init=None):
    in_dim = x.get_shape().as_list()[-1]
    shape = np.array([in_dim, hdim])
    W, decay_term = weight_init(shape, w_init)
    b = tf.Variable(tf.constant(1., shape=[hdim]))
    h = tf.matmul(x, W)
    h = h + b
    return h, decay_term, W, b

def conv_layer(x, out_channel, w_init=None, b_init=None, ksize=3, strides=1):
    in_channel = x.get_shape().as_list()[-1]
    shape = [ksize, ksize, in_channel, out_channel]
    W, decay_term = weight_init(shape, w_init)
    b = tf.Variable(tf.constant(1., shape=[out_channel]))
    h = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    h = tf.nn.bias_add(h, b)
    decay_term = tf.nn.l2_loss(W)
    return h, decay_term, W, b

def max_pool(x, ksize, strides=2):
    return tf.nn.max_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding="SAME")

def avg_pool(x, ksize, strides=2):
    return tf.nn.avg_pool(x, ksize=[1, ksize, ksize, 1], strides=[1, strides, strides, 1], padding="SAME")

def dropout(x, keep_prob, is_train):
    if is_train:
        x = tf.nn.dropout(x, keep_prob)
    return x

def bn_layer(x, is_train, decay=0.9):
    a_shape = x.get_shape().as_list()
    a_num = a_shape[-1]
    gamma = tf.Variable(tf.constant(1., shape=[a_num]))
    beta = tf.Variable(tf.constant(0., shape=[a_num]))
    ema_mean = tf.Variable(tf.constant(1., shape=[a_num]), trainable=False)
    ema_variance = tf.Variable(tf.constant(0., shape=[a_num]), trainable=False)
    if is_train:
        if len(a_shape) == 4:
            bathch_mean, batch_variance = tf.nn.moments(x, [0, 1, 2], keep_dims=False)
        elif len(a_shape) == 2:
            bathch_mean, batch_variance = tf.nn.moments(x, [0], keep_dims=False)
        else:
            raise TypeError("Unrecognized shape %s"%str(a_shape))
        ema_mean_acc = tf.assign(ema_mean, ema_mean*decay + bathch_mean*(1-decay))
        ema_variance_acc = tf.assign(ema_variance, ema_variance*decay + batch_variance*(1-decay))
        with tf.control_dependencies([ema_mean_acc, ema_variance_acc]):
            bn = tf.nn.batch_normalization(x, bathch_mean, batch_variance, beta, gamma, 1e-5)
        return bn, gamma, beta, ema_mean, ema_variance
    else:
        bn = tf.nn.batch_normalization(x, ema_mean, ema_variance, beta, gamma, 1e-5)
        return bn, gamma, beta, ema_mean, ema_variance