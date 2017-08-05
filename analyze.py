import os
import pickle
import util

import numpy as np
import tensorflow as tf
import sklearn as sk
from sklearn.manifold import TSNE
from tensorflow.contrib.tensorboard.plugins import projector

INPUT_PATH = 'data/processed/*.txt'
LOG_PATH = 'data/temp'
STOPWORDS_PATH = 'data/source/stopwords/stopwords.txt'
CHECK_WORDS = [
    'mind', 'body', 'dhamma', 'buddha', 'sangha', 'tathagata', 'gotama', 'right', 'wrong', 'view', 'resolve', 'speech',
    'action', 'livelihood', 'effort', 'mindfulness', 'practice', 'cessation', 'origination', 'birth', 'death',
    'pleasure', 'pain', 'good', 'bad', 'mind', 'body', 'north', 'south', 'skin', 'flesh', 'eye', 'ear', 'king',
    'contact', 'form', 'sensation', 'feeling', 'perception', 'feeling', 'formation', 'consciousness'
]

class NGramScanner(object):

    def __init__(self, sentences, window=5):
        self.sentences = sentences
        self.window = window
        self.ngrams = []
        self.index = 0

    def _extract(self, sentence):

        ngrams = []

        for i in range(len(sentence)):
            start = max(i - self.window, 0)
            end = min(i + self.window, len(sentence) - 1)

            for j in range(start, end + 1):
                if j != i:
                    ngrams.append((sentence[i], sentence[j]))

        return ngrams

    def batch(self, size):

        batch = []
        labels = []

        while len(batch) < size:

            while not self.ngrams:
                self.ngrams = self._extract(self.sentences[self.index])
                self.index = (self.index + 1) % len(self.sentences)

            ngram = self.ngrams.pop(0)
            batch.append(ngram[0])
            labels.append(ngram[1])

        batch = np.array(batch)
        labels = np.array([[l] for l in labels])

        return batch, labels


class Word2Vec(object):

    def __init__(self, n_words, n_embeddings=128, n_batch=128, n_sampled=12):

        self.train_examples = tf.placeholder(tf.int32, shape=[n_batch], name='inputs')
        self.train_labels = tf.placeholder(tf.int32, shape=[n_batch, 1], name='labels')

        embeddings = tf.Variable(tf.random_uniform([n_words, n_embeddings]), name='embeddings')
        train_embeddings = tf.nn.embedding_lookup(embeddings, self.train_examples)

        nce_weights = tf.Variable(tf.truncated_normal([n_words, n_embeddings]), name='nce_weights')
        nce_biases = tf.Variable(tf.random_normal([n_words]), name='nce_biases')

        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=self.train_labels,
                inputs=train_embeddings,
                num_sampled=n_sampled,
                num_classes=n_words))

        self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)

        self.embeddings = tf.nn.l2_normalize(embeddings, 1, name='normalized_embeddings')
        self.similarity = tf.matmul(self.embeddings, self.embeddings, transpose_b=True)

    def train(self, session, inputs, labels):

        feed_dict = {self.train_examples: inputs, self.train_labels: labels}
        _, loss = session.run([self.optimizer, self.loss], feed_dict=feed_dict)

        return loss


def train():

    window = 8
    n_batch = 128
    n_steps = 200001

    with open(STOPWORDS_PATH, 'r') as file:
        stopwords = file.read().strip().split('\n')

    sentences = util.read_data(INPUT_PATH)
    encoded, dictionary, reverse_dictionary = util.encode_data(sentences, ignore=stopwords)
    scanner = NGramScanner(encoded, 8)
    n_words = len(dictionary)

    print('# words:', n_words)

    word2vec = Word2Vec(n_words)
    saver = tf.train.Saver()

    with tf.Session() as session:

        writer = tf.summary.FileWriter(LOG_PATH, session.graph)
        init = tf.global_variables_initializer()
        init.run()

        average_loss = 0

        for step in range(1, n_steps):

            batch, labels = scanner.batch(n_batch)
            average_loss += word2vec.train(session, batch, labels)

            if step % 2000 == 0:
                print('average loss at step {}: {}'.format(step, average_loss / 2000))
                saver.save(session, os.path.join(LOG_PATH, 'model'), step)
                average_loss = 0

            if step % 10000 == 0:
                similarity = word2vec.similarity.eval()#(session)

                for word in CHECK_WORDS:
                    if word in dictionary:
                        nearest = (-similarity[dictionary[word], :]).argsort()[1:9]
                        decoded = [reverse_dictionary[j] for j in nearest]
                        print('  {} -> {}'.format(word, decoded))

        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)

        data = 'label\n'
        data += '\n'.join([reverse_dictionary[i] for i in range(n_words)])
        outpath = os.path.join(LOG_PATH, 'metadata.tsv')

        with open(outpath, 'w') as outfile:
            outfile.write(data)
            print('wrote metadata -> {}'.format(outpath))

        outpath = os.path.join(LOG_PATH, 'dictionary')
        with open(outpath, 'wb') as outfile:
            pickle.dump(dictionary, outfile)
            print('wrote dictionary -> {}'.format(outpath))

        outpath = saver.save(session, os.path.join(LOG_PATH, 'model'))
        print('wrote model -> {}'.format(outpath))

        writer.close()


def evaluate():

    tf.reset_default_graph()

    tsne = TSNE(n_components=2, init='pca', verbose=1)

    with tf.Session() as session:

        with open('data/temp/dictionary', 'rb') as infile:
            dictionary = pickle.load(infile)
            reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        n_words = len(dictionary)

        word2vec = Word2Vec(n_words)
        saver = tf.train.Saver()
        saver.restore(session, os.path.join(LOG_PATH, 'model'))

        embeddings = word2vec.embeddings.eval()
        #sim = word2vec.similarity.eval()

        plot_only = 1000
        labels = [reverse_dictionary[i] for i in range(plot_only)]

        points = tsne.fit_transform(embeddings[:plot_only, :])
        util.plot_points(points, labels, filename='tsne.png')


if __name__ == '__main__':

    saved = os.path.isfile('data/temp/model.index')

    if saved:
        evaluate()
    else:
        train()
        evaluate()
