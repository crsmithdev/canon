import tensorflow as tf
import math


class WordVectors(object):

    def __init__(self, n_words, n_embeddings=100, n_batch=100, n_sampled=5):

        self.train_examples = tf.placeholder(tf.int64, shape=[n_batch], name='inputs')
        self.train_labels = tf.placeholder(tf.int64, shape=[n_batch, 1], name='labels')
        init = 1.0 / (n_embeddings)

        embeddings = tf.Variable(tf.random_uniform([n_words, n_embeddings], -init, init), name='embeddings')
        train_embeddings = tf.nn.embedding_lookup(embeddings, self.train_examples)

        nce_weights = tf.Variable(tf.truncated_normal([n_words, n_embeddings],
            stddev=1.0 / math.sqrt(n_embeddings)), name='nce_weights')
        nce_biases = tf.Variable(tf.zeros([n_words]), name='nce_biases')

        sampled, true_expected, sampled_expected = tf.nn.uniform_candidate_sampler(
            true_classes=self.train_labels,
            num_true=1,
            num_sampled=n_sampled,
            unique=True,
            range_max=n_words,
            name='sampler')

        self.loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=self.train_labels,
                inputs=train_embeddings,
                num_sampled=n_sampled,
                num_classes=n_words,
                sampled_values=(sampled, true_expected, sampled_expected)))

        self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

        self.embeddings = tf.nn.l2_normalize(embeddings, 1, name='normalized_embeddings')
        self.similarity = tf.matmul(self.embeddings, self.embeddings, transpose_b=True)

    def train(self, session, inputs, labels):

        feed_dict = {self.train_examples: inputs, self.train_labels: labels}
        _, loss = session.run([self.optimizer, self.loss], feed_dict=feed_dict)

        return loss
