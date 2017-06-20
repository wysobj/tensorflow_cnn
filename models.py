import os
import numpy as np
import tensorflow as tf
import pickle
# from matplotlib import pyplot as plt
import functions

class Classifier(object):

    def __init__(self,
                      img_shape, 
                      classes_num, 
                      sess, 
                      model_id,
                      optimizer=None,
                      batch_size=64, 
                      max_epoch=100, 
                      patience=25,
                      eval_num=2000,
                      is_train=True):
        self.img_shape = img_shape
        self.classes_num = classes_num
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.sess = sess
        self.eval_num = eval_num
        self.is_train = is_train
        self.bns = {}
        self.__build__()
        if optimizer != None:
            self.train_op = optimizer.minimize(self.loss)
        self.epoch = 0
        self.sess.run(tf.global_variables_initializer())
        self.model_id = model_id

    def __build__(self, init_params):
        pass

    def fit(self, X, y, keep_prob=0.5):
        data_size = len(X)
        indexes = np.arange(data_size)
        batches = int(len(indexes) / self.batch_size)
        np.random.shuffle(indexes)
        for i in range(batches):
            start_idx = self.batch_size * i
            end_idx = self.batch_size * (i+1)
            if i == batches - 1:
                end_idx = -1
            batch_x = X[indexes[start_idx : end_idx]]
            batch_y = y[indexes[start_idx : end_idx]]
            _, loss = self._fit(batch_x, batch_y)
            print("Epoch %d, batch %d/%d, loss %f"%(self.epoch, i+1, batches, loss))
        self.epoch += 1

    def _fit(self, x, y):
        update, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.x : x, self.y : y})
        return update, loss

    def predict(self, X):
        predicts = self.sess.run(self.predict, feed_dict={self.x : X}).argmax(axis=1)
        return predicts

    def distribution(self, X):
        probabilities = self.sess.run(self.predict, feed_dict={self.x : X})
        return probabilities

    def eval_loss(self, X, y):
        loss = self.sess.run(self.loss, feed_dict={self.x : X, self.y : y})
        return loss

    def eval_h(self, X, h_id):
        h = self.hiddens.get(h_id, None)
        if h == None:
            print("Unknown hidden layer id.")
            return
        hd = self.sess.run(h, feed_dict={x : X})
        return hd

    def get_trainable_variables(self):
        variables = {}
        variables.update(self.weights)
        variables.update(self.biases)
        variables.update(self.bns)
        return variables


class ConvNet1(Classifier):
    
    def __build__(self, weight_decay=1e-4):
        inputs_shape = [None]
        inputs_shape.extend(self.img_shape)
        x = tf.placeholder(dtype=tf.float32, shape=inputs_shape)
        labels_shape = [None, self.classes_num]
        y = tf.placeholder(dtype=tf.int32, shape=labels_shape)
        h1, dt1, w1, b1 = functions.conv_layer(x, 64, ksize=3)
        h1 = tf.nn.relu(h1)
        h1 = functions.max_pool(h1, ksize=3, strides=2)
        h2, dt2, w2, b2 = functions.conv_layer(h1, 128, ksize=3)
        h2 = tf.nn.relu(h2)
        h2 = functions.max_pool(h2, ksize=3, strides=2)
        h3, dt3, w3, b3 = functions.conv_layer(h2, 256, ksize=3)
        h3 = tf.nn.relu(h3)
        h3 = functions.max_pool(h3, ksize=3, strides=2)
        flatten = tf.reshape(h3, [-1, functions.dim(h3)])
        h4, dt4, w4, b4 = functions.dense_layer(flatten, 1024)
        h4 = tf.nn.relu(h4)
        h5, dt5, w5, b5 = functions.dense_layer(h4, self.classes_num)
        predict = tf.nn.softmax(h5)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h5, labels=y))
        if weight_decay:
            loss += weight_decay * (dt1 + dt2 + dt3 + dt4 + dt5)
        self.loss = loss
        self.predict = predict
        self.weights = {"W1" : w1, "W2" : w2, "W3" : w3, "W4" : w4, "W5" : w5}
        self.biases = {"b1" : b1, "b2" : b2, "b3" : b3, "b4" : b4, "b5" : b5}
        self.hiddens = {"h1" : h1, "h2" : h2, "h3" : h3, "h4" : h4, "h5" : h5}
        self.x = x
        self.y = y


class ConvNet2(Classifier):

    def __build__(self, weight_decay=1e-4):
        keep_prob = tf.placeholder(dtype=tf.float32)
        inputs_shape = [None]
        inputs_shape.extend(self.img_shape)
        x = tf.placeholder(dtype=tf.float32, shape=inputs_shape)
        labels_shape = [None, self.classes_num]
        y = tf.placeholder(dtype=tf.int32, shape=labels_shape)
        h1, dt1, w1, b1 = functions.conv_layer(x, 64, ksize=3)
        h1 = tf.nn.relu(h1)
        h1 = functions.max_pool(h1, ksize=3, strides=2)
        h2, dt2, w2, b2 = functions.conv_layer(h1, 128, ksize=3)
        h2 = tf.nn.relu(h2)
        h2 = functions.max_pool(h2, ksize=3, strides=2)
        h3, dt3, w3, b3 = functions.conv_layer(h2, 256, ksize=3)
        h3 = tf.nn.relu(h3)
        h3 = functions.max_pool(h3, ksize=3, strides=2)
        flatten = tf.reshape(h3, [-1, functions.dim(h3)])
        h4, dt4, w4, b4 = functions.dense_layer(flatten, 1024)
        h4 = tf.nn.relu(h4)
        h4 = tf.nn.dropout(h4, keep_prob)
        h5, dt5, w5, b5 = functions.dense_layer(h4, self.classes_num)
        predict = tf.nn.softmax(h5)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h5, labels=y))
        if weight_decay:
            loss += weight_decay * (dt1 + dt2 + dt3 + dt4 + dt5)
        self.loss = loss
        self.predict = predict
        self.weights = {"W1" : w1, "W2" : w2, "W3" : w3, "W4" : w4, "W5" : w5}
        self.biases = {"b1" : b1, "b2" : b2, "b3" : b3, "b4" : b4, "b5" : b5}
        self.hiddens = {"h1" : h1, "h2" : h2, "h3" : h3, "h4" : h4, "h5" : h5}
        self.x = x
        self.y = y
        self.keep_prob = keep_prob

    def _fit(self, x, y):
        update, loss = self.sess.run([self.train_op, self.loss], feed_dict={self.x : x, self.y : y, self.keep_prob : 0.5})
        return update, loss

    def predict(self, X):
        predicts = self.sess.run(self.predict, feed_dict={self.x : X, self.keep_prob : 1.}).argmax(axis=1)
        return predicts

    def distribution(self, X):
        probabilities = self.sess.run(self.predict, feed_dict={self.x : X, self.keep_prob : 1.})
        return probabilities

    def eval_loss(self, X, y):
        loss = self.sess.run(self.loss, feed_dict={self.x : X, self.y : y, self.keep_prob : 1.})
        return loss


class ConvNet3(Classifier):

    def __build__(self, weight_decay=1e-4):
        inputs_shape = [None]
        inputs_shape.extend(self.img_shape)
        x = tf.placeholder(dtype=tf.float32, shape=inputs_shape)
        labels_shape = [None, self.classes_num]
        y = tf.placeholder(dtype=tf.int32, shape=labels_shape)
        h1, dt1, w1, b1 = functions.conv_layer(x, 64, ksize=3)
        h1, gamma1, beta1, ema_mean1, ema_variance1 = functions.bn_layer(h1, self.is_train)
        h1 = tf.nn.relu(h1)
        h1 = functions.max_pool(h1, ksize=3, strides=2)
        h2, dt2, w2, b2 = functions.conv_layer(h1, 128, ksize=3)
        h2, gamma2, beta2, ema_mean2, ema_variance2 = functions.bn_layer(h2, self.is_train)
        h2 = tf.nn.relu(h2)
        h2 = functions.max_pool(h2, ksize=3, strides=2)
        h3, dt3, w3, b3 = functions.conv_layer(h2, 256, ksize=3)
        h3, gamma3, beta3, ema_mean3, ema_variance3 = functions.bn_layer(h3, self.is_train)
        h3 = tf.nn.relu(h3)
        h3 = functions.max_pool(h3, ksize=3, strides=2)
        flatten = tf.reshape(h3, [-1, functions.dim(h3)])
        h4, dt4, w4, b4 = functions.dense_layer(flatten, 1024)
        h4, gamma4, beta4, ema_mean4, ema_variance4 = functions.bn_layer(h4, self.is_train)
        h4 = tf.nn.relu(h4)
        h5, dt5, w5, b5 = functions.dense_layer(h4, self.classes_num)
        predict = tf.nn.softmax(h5)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=h5, labels=y))
        if weight_decay:
            loss += weight_decay * (dt1 + dt2 + dt3 + dt4 + dt5)
        self.loss = loss
        self.predict = predict
        self.weights = {"W1" : w1, "W2" : w2, "W3" : w3, "W4" : w4, "W5" : w5}
        self.biases = {"b1" : b1, "b2" : b2, "b3" : b3, "b4" : b4, "b5" : b5}
        self.hiddens = {"h1" : h1, "h2" : h2, "h3" : h3, "h4" : h4, "h5" : h5}
        self.bns = {"gamma1" : gamma1, "gamma2" : gamma2, "gamma3" : gamma3, "gamma4" : gamma4,
                            "beta1" : beta1, "beta2" : beta2, "beta3" : beta3, "beta4" : beta4, "ema_mean1" : ema_mean1, "ema_mean2" : ema_mean2,
                            "ema_mean3" : ema_mean3, "ema_mean4" : ema_mean4, "ema_variance1" : ema_variance1, "ema_variance2" : ema_variance2,
                            "ema_variance3" : ema_variance3, "ema_variance4" : ema_variance4}
        self.x = x
        self.y = y