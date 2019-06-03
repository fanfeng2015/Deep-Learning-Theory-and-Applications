from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

BATCH_SIZE = 50
NUM_STEPS = 30000

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def fc_layer(previous, input_size, output_size):
    W = weight_variable([input_size, output_size])
    b = bias_variable([output_size])
    return tf.matmul(previous, W) + b

def autoencoder(x):
    hidden = tf.nn.relu(fc_layer(x, 28*28, 2))                # hidden layer
    output = tf.nn.relu(fc_layer(hidden, 2, 28*28))           # output layer (ReLU)
    loss = tf.reduce_mean(tf.squared_difference(x, output))   # MSE loss on output
    return loss, output, hidden

def main():
    mnist = input_data.read_data_sets('/tmp/MNIST_data')
    x = tf.placeholder(tf.float32, shape=[None, 784])
    loss, output, latent = autoencoder(x)
    optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1, NUM_STEPS + 1):
            batch = mnist.train.next_batch(BATCH_SIZE)
            feed = {x : batch[0]}
            _, temp = sess.run([optimizer, loss], feed_dict=feed)
            if i % 1000 == 0:
                print("Step %d, mini-batch loss: %g" % (i, temp))

        # latent space
        pred = sess.run(latent, feed_dict={x : mnist.test._images})
        pred = np.asarray(pred)
        pred = np.reshape(pred, (mnist.test._num_examples, 2))

        plt.figure(figsize=(10, 10))
        plt.scatter(pred[:, 0], pred[:, 1], c=mnist.test._labels,cmap='brg')
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    main()


