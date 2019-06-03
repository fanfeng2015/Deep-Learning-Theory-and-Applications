import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sklearn.decomposition
import re
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

BATCH_SIZE = 10
ITERATIONS = 1000
SEQ_LENGTH = 50
EMBEDDING_SIZE = 100
LSTM_SIZE = 64   # number of hiddent units per lstm cell 
LSTM_LAYERS = 2
# TODO 1: put a text file of your choosing in the same directory and put its name here
TEXT_FILE = 'input.txt'

string = open(TEXT_FILE).read()
tokens = re.split('\W+', string)
vocabulary = sorted(set(tokens))
VOCABULARY_SIZE = len(vocabulary)

tf.reset_default_graph()
x = tf.placeholder(tf.int32, [None, None], name='x')

# TODO 2: create variable for embedding matrix and do an embedding_lookup for x
embedding = tf.Variable(tf.random_uniform([VOCABULARY_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
embedding_lookup_for_x = tf.nn.embedding_lookup(embedding, x)

def lstm_cell():
   lstm = tf.nn.rnn_cell.LSTMCell(LSTM_SIZE, reuse=tf.get_variable_scope().reuse)
   return lstm

# TODO 3: define an lstm encoder function that takes the embedding lookup and produces a final state
lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell() for _ in range(LSTM_LAYERS)])
init_state = lstm_cell.zero_state(BATCH_SIZE, tf.float32)
_, final_state = tf.nn.dynamic_rnn(lstm_cell, embedding_lookup_for_x, initial_state=init_state)

# # TODO 4: define an lstm decoder function that takes the final state from previous step and produces a sequence of outputs
outs, _ = tf.nn.dynamic_rnn(lstm_cell, embedding_lookup_for_x, initial_state=final_state)
outs = tf.layers.dense(outs, VOCABULARY_SIZE)

# create loss/train ops
one_hots = tf.one_hot(x, VOCABULARY_SIZE)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outs, labels=one_hots))
opt = tf.train.AdamOptimizer(.001)
train_op = opt.minimize(loss)

# create a session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# do training
i = 0
for num_iter in range(ITERATIONS):
    if num_iter % 10 == 0: print(num_iter)
    batch = [[vocabulary.index(v) for v in tokens[ii:ii + SEQ_LENGTH]] for ii in range(i, i + BATCH_SIZE)]
    batch = np.stack(batch, axis=0)
    i += BATCH_SIZE
    if i + BATCH_SIZE + SEQ_LENGTH > len(tokens): i = 0
    sess.run(train_op, feed_dict={x: batch})

# plot word embeddings
fig = plt.figure()
learned_embeddings = sess.run(embedding)
learned_embeddings_pca = sklearn.decomposition.PCA(2).fit_transform(learned_embeddings)
ax = fig.add_subplot(1, 1, 1)
ax.scatter(learned_embeddings_pca[:, 0], learned_embeddings_pca[:, 1], s=5, c='w')
MIN_SEPARATION = .1 * min(ax.get_xlim()[1] - ax.get_xlim()[0], ax.get_ylim()[1] - ax.get_ylim()[0])

fig.clf()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(learned_embeddings_pca[:, 0], learned_embeddings_pca[:, 1], s=5, c='w')

# TODO 5: run this multiple times
xy_plotted = set()
for i in np.random.choice(VOCABULARY_SIZE, VOCABULARY_SIZE, replace=False):
    x_, y_ = learned_embeddings_pca[i]
    if any([(x_ - point[0])**2 + (y_ - point[1])**2 < MIN_SEPARATION for point in xy_plotted]): continue
    xy_plotted.add(tuple([learned_embeddings_pca[i, 0], learned_embeddings_pca[i, 1]]))
    ax.annotate(vocabulary[i], xy=learned_embeddings_pca[i])

plt.show()


