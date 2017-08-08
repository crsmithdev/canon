import os
import pickle
import util

import numpy as np
import tensorflow as tf
import sklearn as sk
from sklearn.manifold import TSNE
from tensorflow.contrib.tensorboard.plugins import projector
import collections

INPUT_PATH = 'data/processed/*.txt'
LOG_PATH = 'data/temp'
STOPWORDS_PATH = 'data/source/stopwords/stopwords.txt'
CHECK_WORDS = [
    'mind', 'body', 'dhamma', 'buddha', 'sangha', 'tathagata', 'gotama', 'ananda', 'right', 'wrong', 'view', 'resolve',
    'speech', 'action', 'livelihood', 'effort', 'mindfulness', 'practice', 'cessation', 'origination', 'birth', 'death',
    'pleasure', 'pain', 'good', 'bad', 'mind', 'body', 'north', 'south', 'skin', 'flesh', 'eye', 'ear', 'king', 'sword',
    'contact', 'form', 'sensation', 'feeling', 'perception', 'feeling', 'formation', 'consciousness'
]


class NGramScanner(object):

    def __init__(self, sentences, counter, total_words, reverse_dictionary, window=5):
        #self.sentences = sentences
        self.window = window
        self.ngrams = []
        #self.counter = counter
        #self.total_words = total_words
        #self.reverse_dictionary = reverse_dictionary
        #self.dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))
        self.index = 0

        #orig = []
        #for s in sentences:
        #    orig.append([self.reverse_dictionary[w] for w in s])

        counter = collections.Counter()
        total_words = 0

        orig = sentences
        for sentence in orig:
            counter.update(sentence)
            total_words += len(sentence)

        dictionary = {c[0]: i for i, c in enumerate(counter.most_common())}
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

        encoded = []
        for s in orig:
            encoded.append([dictionary[w] for w in s])

        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self.total_words = total_words
        self.counter = counter
        self.sentences = encoded

    def _extract(self, sentence):

        ngrams = []
        cleaned = []

        for word_id in sentence:

            word = self.reverse_dictionary[word_id]

            if self.counter[word] < 5:
                #print('  [skip] {} ({})'.format(word, self.counter[word]))
                continue

            p = self.counter[word] / self.total_words
            r = (math.sqrt(p / .001) + 1) * (.001 / p)

            if np.random.sample() > r:
                #print('  [sampled out] {} ({})'.format(word, r))
                continue

            cleaned.append(word_id)

        sentence = cleaned

        for i, word_id in enumerate(sentence):

            window = np.random.randint(low=1, high=self.window + 1)
            start = max(i - window, 0)
            #start = max(i, 0)
            end = min(i + window, len(sentence) - 1)

            for j in range(start, end + 1):
                if j == i:
                    continue

                pair_id = sentence[j]

                ngrams.append((word_id, pair_id))

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


import math


class Word2Vec(object):

    def __init__(self, n_words, n_embeddings=256, n_batch=256, n_sampled=5):

        self.train_examples = tf.placeholder(tf.int64, shape=[n_batch], name='inputs')
        self.train_labels = tf.placeholder(tf.int64, shape=[n_batch, 1], name='labels')
        v = 1.0 / (n_embeddings)
        #v = 1.0

        embeddings = tf.Variable(tf.random_uniform([n_words, n_embeddings], -v, v), name='embeddings')
        train_embeddings = tf.nn.embedding_lookup(embeddings, self.train_examples)

        #nce_weights = tf.Variable(tf.truncated_normal([n_words, n_embeddings],
        #    stddev=1.0 / math.sqrt(n_embeddings)), name='nce_weights')
        nce_weights = tf.Variable(tf.random_uniform([n_words, n_embeddings], -v, v))
        nce_biases = tf.Variable(tf.zeros([n_words]), name='nce_biases')

        sampled, true_expected, sampled_expected = tf.nn.uniform_candidate_sampler(
            true_classes=self.train_labels,
            num_true=1,
            num_sampled=n_sampled,
            unique=True,
            range_max=n_words,
            name='sampler')

        # self.loss = tf.reduce_mean(
        #     tf.nn.nce_loss(
        #         weights=nce_weights,
        #         biases=nce_biases,
        #         labels=self.train_labels,
        #         inputs=train_embeddings,
        #         num_sampled=n_sampled,
        #         remove_accidental_hits=True,
        #         sampled_values=(sampled, true_expected, sampled_expected),
        #         num_classes=n_words))

        self.loss = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(
                weights=nce_weights,
                biases=nce_biases,
                labels=self.train_labels,
                inputs=train_embeddings,
                num_sampled=n_sampled,
                num_classes=n_words,
                sampled_values=(sampled, true_expected, sampled_expected)))

        reg = tf.reduce_sum(tf.abs(nce_weights))

        self.optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(self.loss)
        #self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)
        #self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.embeddings = tf.nn.l2_normalize(embeddings, 1, name='normalized_embeddings')
        self.similarity = tf.matmul(self.embeddings, self.embeddings, transpose_b=True)

    def train(self, session, inputs, labels):

        feed_dict = {self.train_examples: inputs, self.train_labels: labels}
        _, loss = session.run([self.optimizer, self.loss], feed_dict=feed_dict)

        return loss


def train():

    n_batch = 200
    n_embeddings = 100
    n_steps = 100001
    window = 5

    with open(STOPWORDS_PATH, 'r') as file:
        stopwords = file.read().strip().split('\n')

    sentences = util.read_data(INPUT_PATH)
    encoded, counter, total_words, dictionary, reverse_dictionary = util.encode_data(sentences, ignore=stopwords)
    scanner = NGramScanner(encoded, counter, total_words, reverse_dictionary, window=window)
    n_words = len(dictionary)

    print('# words:', n_words)

    word2vec = Word2Vec(n_words, n_embeddings=n_embeddings, n_batch=n_batch)
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
                print(scanner.index)

            if step % 10000 == 0:
                print(word2vec.similarity)
                similarity = word2vec.similarity.eval()

                for word in CHECK_WORDS:
                    if word in dictionary:
                        nearest = (-similarity[dictionary[word], :]).argsort()[1:9]
                        decoded = [reverse_dictionary[j] for j in nearest]
                        print('  {} -> {}'.format(word, decoded))

        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)

        data = '\n'.join([reverse_dictionary[i] for i in range(n_words)])
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

        with open(os.path.join(LOG_PATH, 'sentences.txt'), 'w') as outfile:
            for sentence in sentences:
                outfile.write(' '.join(sentence) + '\n')

        import word2vec as w2v
        w2v.word2vec('data/temp/sentences.txt', 'data/temp/word2vec.bin', size=100, verbose=True)


def evaluate():

    tf.reset_default_graph()

    with open('data/temp/dictionary', 'rb') as infile:
        dictionary = pickle.load(infile)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    tsne = TSNE(n_components=2, init='pca', verbose=1)

    n_words = len(dictionary)
    #word2vec = Word2Vec(n_words)
    word2vec = Word2Vec(n_words, n_embeddings=100, n_batch=200)
    saver = tf.train.Saver()

    with tf.Session() as session:

        print(n_words)

        saver.restore(session, os.path.join(LOG_PATH, 'model'))

        similarity = word2vec.similarity.eval()
        embeddings = word2vec.embeddings.eval()

        for word in CHECK_WORDS:
            if word in dictionary:
                nearest = (-similarity[dictionary[word], :]).argsort()[1:9]
                decoded = [reverse_dictionary[j] for j in nearest]
                print('  {} -> {}'.format(word, decoded))

        plot_only = 1000
        labels = [reverse_dictionary[i] for i in range(plot_only)]

        points = tsne.fit_transform(embeddings[:plot_only, :])
        util.plot_points(points, labels, filename='tsne.png')

    import word2vec

    sentences = util.read_data(INPUT_PATH)
    encoded, _, _, dictionary, reverse_dictionary = util.encode_data(sentences)
    #scanner = NGramScanner(encoded, 8)
    #n_words = len(dictionary)

    with open(os.path.join(LOG_PATH, 'sentences.txt'), 'w') as outfile:
        for sentence in sentences:
            outfile.write(' '.join(sentence) + '\n')

    #word2vec.word2vec(
    #    'data/temp/sentences.txt', 'data/temp/word2vec.bin', size=1000, verbose=True)
    model = word2vec.load('data/temp/word2vec.bin')
    #print(model.vectors.shape)

    for word in CHECK_WORDS:
        if word in model.vocab:
            indexes, metrics = model.cosine(word)
            closest = [c[0] for c in model.generate_response(indexes, metrics)]
            print('  {} -> {}'.format(word, closest))

    top = [i for i in range(1000)]
    top = [reverse_dictionary[i] for i in top]
    selected = [model[t] for t in top]
    #print(selected, len(selected))
    points = tsne.fit_transform(selected)
    util.plot_points(points, top, filename='tsne.png')


if __name__ == '__main__':

    if os.path.isfile('data/temp/model.index'):
        evaluate()
    else:
        train()
        evaluate()
