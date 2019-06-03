from tensorflow.examples.tutorials.mnist import input_data

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

BATCH_SIZE = 50
NUM_STEPS = 30000
NUM_EXAMPLES = 10

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

def autoencoder(x, num_neurons):
    hidden = tf.nn.relu(fc_layer(x, 28*28, num_neurons))        # hidden layer
    output = tf.nn.relu(fc_layer(hidden, num_neurons, 28*28))   # output layer (ReLU)
    loss = tf.reduce_mean(tf.squared_difference(x, output))     # MSE loss on output
    return loss, output, hidden

def main():
    mnist = input_data.read_data_sets('/tmp/MNIST_data')
    x = tf.placeholder(tf.float32, shape=[None, 784])

    num_neurons_range = [4, 8, 16]
    orig, _ = mnist.test.next_batch(NUM_EXAMPLES)               # test set
    orig_recon = np.empty((28*(len(num_neurons_range)+1), 28*NUM_EXAMPLES))
    for j in range(NUM_EXAMPLES):
        orig_recon[0:28, j*28:(j+1)*28] = orig[j, :].reshape(28, 28)

    for i in range(len(num_neurons_range)):
        num_neurons = num_neurons_range[i]
        print("\nNumber of neurons in hidden layer: %d \n" % (num_neurons))
        loss, output, latent = autoencoder(x, num_neurons)
        optimizer = tf.train.AdamOptimizer(1e-4).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for j in range(1, NUM_STEPS + 1):
                batch = mnist.train.next_batch(BATCH_SIZE)
                feed = {x: batch[0]}
                _, temp = sess.run([optimizer, loss], feed_dict=feed)
                if j % 1000 == 0:
                    print("Step %d, mini-batch loss: %g" % (j, temp))

            # reconstruct and plot
            recon = sess.run(output, feed_dict={x: orig})
            for j in range(NUM_EXAMPLES):
                orig_recon[28*(i+1):28*(i+2), j*28:(j+1)*28] = recon[j, :].reshape(28, 28)

    plt.figure(figsize=(8, 40))
    plt.imshow(orig_recon, cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()


