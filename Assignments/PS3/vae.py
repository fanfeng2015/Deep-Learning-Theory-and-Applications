# import MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("tmp/", one_hot=False)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# set params
batch_size = 32
input_dim = 784
num_iter = 10000

class VAE(object):
    def __init__(self, zdim=16, lr=.0001, batch_size=64):
        self.lr = lr
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.zdim = zdim
        self.build()

        # TODO: define a session here
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def encoder(self, x):
        with tf.variable_scope('encoder'):
            layer1 = tf.layers.dense(x, 256, name='layer1', activation=tf.nn.relu)
            layer2 = tf.layers.dense(layer1, 128, name='layer2', activation=tf.nn.relu)
            layer3 = tf.layers.dense(layer2, 64, name='layer3', activation=tf.nn.relu)

        mu = tf.layers.dense(layer3, self.zdim, name='mu')
        logsigma = tf.layers.dense(layer3, self.zdim, name='logsigma')

        return mu, logsigma

    def decoder(self, z):
        with tf.variable_scope('decoder'):
            layer1 = tf.layers.dense(z, 64, name='layer1', activation=tf.nn.relu)
            layer2 = tf.layers.dense(layer1, 128, name='layer2', activation=tf.nn.relu)
            layer3 = tf.layers.dense(layer2, 256, name='layer3', activation=tf.nn.relu)

        xhat = tf.layers.dense(layer3, self.input_dim, name='layer4', activation=tf.sigmoid)

        return xhat

    def build(self):
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, self.input_dim])

        # encoder
        mu, logsigma = self.encoder(self.x)

        # sample from random normal
        eps = tf.random_normal(shape=tf.shape(logsigma), mean=0, stddev=1)
        # reparameterization trick
        z = mu + tf.sqrt(tf.exp(logsigma)) * eps

        # decoder
        self.xhat = self.decoder(z)

        # TODO: define reconstruction loss here
        epsilon = 1e-10
        recon_loss = -tf.reduce_sum(self.x * tf.log(epsilon + self.xhat) + (1 - self.x) * tf.log(epsilon + 1 - self.xhat), axis=1)

        # TODO: define kl loss here
        kl_loss = -0.5 * tf.reduce_sum(1 + logsigma - tf.square(mu) - tf.exp(logsigma), axis=1)

        self.loss_op = tf.reduce_mean(recon_loss) + tf.reduce_mean(kl_loss)
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss_op)

    def train(self, x):
        self.sess.run(self.train_op, feed_dict={self.x: x})

    def get_reconstruction(self, x):
        xhat = self.sess.run(self.xhat, feed_dict={self.x: x})
        return xhat

vae = VAE()

for i in range(num_iter):
    if i % 100 == 0: print("Iter {} / {}".format(i, num_iter))
    vae.train(mnist.train.next_batch(batch_size)[0])

# TODO: plotting code
n = 10
input_data = mnist.train.next_batch(n)
recon_data = vae.get_reconstruction(input_data[0])

orig_recon = np.empty((28*2, n * 28))
for i in range(n):
    orig_recon[0:28, i*28:(i+1)*28] = input_data[0][i, :].reshape(28, 28)
    orig_recon[28:28*2, i*28:(i+1)*28] = recon_data[i, :].reshape(28, 28)

plt.figure(figsize=(8, 40))
plt.imshow(orig_recon, cmap='gray')
plt.show()


