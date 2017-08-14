import math
import tensorflow as tf


class WordVectors(object):
    """Word vector model."""

    def __init__(self,
                 vocabulary_size,
                 embedding_size=100,
                 n_sampled=5,
                 learning_rate=0.3):

        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.n_sampled = n_sampled
        self.learning_rate = learning_rate

        self._build()

    def _build(self):

        init = 1.0 / math.sqrt(self.embedding_size)  # weight initialization

        # input training examples & labels
        self._train_examples = tf.placeholder(tf.int64, shape=[None], name='inputs')
        self._train_labels = tf.placeholder(tf.int64, shape=[None, 1], name='labels')

        # word vector embeddings
        embeddings = tf.Variable(
            tf.random_uniform([self.vocabulary_size, self.embedding_size], -init, init),
            name='embeddings')
        train_embeddings = tf.nn.embedding_lookup(embeddings, self._train_examples)

        # softmax weights + bias
        softmax_weights = tf.Variable(
            tf.truncated_normal([self.vocabulary_size, self.embedding_size], stddev=init),
            name='softmax_weights')
        softmax_bias = tf.Variable(tf.zeros([self.vocabulary_size]), name='softmax_bias')

        # don't use default log uniform sampler, distribution has been changed by subsampling
        candidates, true_expected, sampled_expected = tf.nn.learned_unigram_candidate_sampler(
            true_classes=self._train_labels,
            num_true=1,
            num_sampled=self.n_sampled,
            unique=True,
            range_max=self.vocabulary_size,
            name='sampler')

        # sampled softmax loss
        self._loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                weights=softmax_weights,
                biases=softmax_bias,
                labels=self._train_labels,
                inputs=train_embeddings,
                num_sampled=self.n_sampled,
                num_classes=self.vocabulary_size,
                sampled_values=(candidates, true_expected, sampled_expected)))

        self._optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(
            self._loss)
        self._loss_summary = tf.summary.scalar('loss', self._loss)

        # inputs for nearest N
        self._nearest_to = tf.placeholder(tf.int64)

        # cosine similarity for nearest N
        self._normalized_embeddings = tf.nn.l2_normalize(embeddings, 1)
        target_embedding = tf.gather(self._normalized_embeddings, self._nearest_to)
        self._distance = tf.matmul(
            target_embedding, self._normalized_embeddings, transpose_b=True)

        self.eval_a = tf.placeholder(tf.int64)
        self.eval_b = tf.placeholder(tf.int64)
        self.eval_c = tf.placeholder(tf.int64)

        a_emb = tf.gather(self._normalized_embeddings, self.eval_a)
        b_emb = tf.gather(self._normalized_embeddings, self.eval_b)
        c_emb = tf.gather(self._normalized_embeddings, self.eval_c)

        target = c_emb + (b_emb - a_emb)
        self.dist2 = tf.matmul(target, self._normalized_embeddings, transpose_b=True)



    def train(self, session, examples, labels):
        """Trains the model on a set of examples and labels."""

        return session.run(
            [self._optimizer, self._loss, self._loss_summary],
            feed_dict={
                self._train_examples: examples,
                self._train_labels: labels,
            })

    def nearest(self, session, inputs, n=8):
        """Returns the nearest neighbors for a set of inputs."""

        _, indices = session.run(
            tf.nn.top_k(self._distance, n + 1), feed_dict={self._nearest_to: inputs})

        return indices

    def analogy(self, session, a, b, c, n=4):

        _, indices = session.run(
            tf.nn.top_k(self.dist2, n + 1), feed_dict={
                self.eval_a: [a],
                self.eval_b: [b],
                self.eval_c: [c],
            })

        return indices

    def embeddings(self, session):
        """Returns the normalized embeddings of the model."""

        return session.run(self._normalized_embeddings)
