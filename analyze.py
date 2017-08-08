import os
import pickle

import numpy as np
import tensorflow as tf
import sklearn as sk
from sklearn.manifold import TSNE
from tensorflow.contrib.tensorboard.plugins import projector
import collections
import model
import skipgram
from matplotlib import pyplot as plt
import glob

INPUT_PATH = 'data/processed/*.txt'
LOG_PATH = 'data/temp'
STOPWORDS_PATH = 'data/source/stopwords/stopwords.txt'
CHECK_WORDS = [
    'mind', 'body', 'dhamma', 'buddha', 'sangha', 'tathagata', 'gotama', 'ananda', 'right', 'wrong', 'view', 'resolve',
    'speech', 'action', 'livelihood', 'effort', 'mindfulness', 'practice', 'cessation', 'origination', 'birth', 'death',
    'pleasure', 'pain', 'good', 'bad', 'mind', 'body', 'north', 'south', 'skin', 'flesh', 'eye', 'ear', 'king', 'sword',
    'contact', 'form', 'sensation', 'feeling', 'perception', 'feeling', 'formation', 'consciousness'
]

N_BATCH = 200
N_EMBEDDINGS = 100
N_SAMPLED = 20
WINDOW_SIZE = 5
N_STEPS = 100001
N_SUBSAMPLING = 1e5
N_PLOT = 1000


def read_data(inpath):

    sentences = []

    for path in glob.iglob(inpath):
        with open(path, 'r') as infile:

            lines = infile.read().split('\n')
            sentences.extend([l.split(' ') for l in lines])

    return sentences


def plot_points(points, labels, filename=None):

    plt.figure(figsize=(64, 64))

    for i, label in enumerate(labels):
        x, y = points[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    if filename:
        plt.savefig(filename)


def train():

    sentences = read_data(INPUT_PATH)
    scanner = skipgram.SkipGramScanner(sentences, window=WINDOW_SIZE, sample=N_SUBSAMPLING)

    dictionary = scanner.dictionary
    reverse_dictionary = scanner.reverse_dictionary
    n_words = len(dictionary)

    print('# words:', n_words)

    word2vec = model.WordVectors(n_words, n_embeddings=N_EMBEDDINGS, n_batch=N_BATCH, n_sampled=N_SAMPLED)
    saver = tf.train.Saver()

    with tf.Session() as session:

        writer = tf.summary.FileWriter(LOG_PATH, session.graph)
        init = tf.global_variables_initializer()
        init.run()

        average_loss = 0

        for step in range(1, N_STEPS):

            batch, labels = scanner.batch(N_BATCH)
            average_loss += word2vec.train(session, batch, labels)

            if step % 2000 == 0:
                print('average loss at step {}: {}'.format(step, average_loss / 2000))
                saver.save(session, os.path.join(LOG_PATH, 'model'), step)
                average_loss = 0
                print(scanner.index)

            if step % 10000 == 0:
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
    word2vec = model.WordVectors(len(dictionary), n_embeddings=N_EMBEDDINGS, n_batch=N_BATCH, n_sampled=N_SAMPLED)
    saver = tf.train.Saver()

    with tf.Session() as session:

        saver.restore(session, os.path.join(LOG_PATH, 'model'))

        similarity = word2vec.similarity.eval()
        embeddings = word2vec.embeddings.eval()

        for word in CHECK_WORDS:
            if word in dictionary:
                nearest = (-similarity[dictionary[word], :]).argsort()[1:9]
                decoded = [reverse_dictionary[j] for j in nearest]
                print('  {} -> {}'.format(word, decoded))

        labels = [reverse_dictionary[i] for i in range(N_PLOT)]
        points = tsne.fit_transform(embeddings[:N_PLOT, :])
        plot_points(points, labels, filename='tsne.png')

    import word2vec

    sentences = read_data(INPUT_PATH)

    with open(os.path.join(LOG_PATH, 'sentences.txt'), 'w') as outfile:
        for sentence in sentences:
            outfile.write(' '.join(sentence) + '\n')

    model2 = word2vec.load('data/temp/word2vec.bin')

    for word in CHECK_WORDS:
        if word in model2.vocab:
            indexes, metrics = model2.cosine(word)
            closest = [c[0] for c in model2.generate_response(indexes, metrics)]
            print('  {} -> {}'.format(word, closest))


if __name__ == '__main__':

    if os.path.isfile('data/temp/model.index'):
        evaluate()
    else:
        train()
        evaluate()
