import os
import tensorflow as tf
from sklearn.manifold import TSNE
import model
import skipgram
from matplotlib import pyplot as plt
import word2vec

INPUT_PATH = 'data/sentences.txt'
LOG_PATH = 'data/model'
BATCH_SIZE = 100
EMBEDDING_SIZE = 200
N_SAMPLED = 10
WINDOW_SIZE = 5
N_STEPS = 200001
N_SUBSAMPLING = 1e3
N_PLOT = 1000
LEARNING_RATE = 0.3
CHECK_WORDS = [
    'dhamma', 'buddha', 'sangha', 'tathagata', 'gotama', 'ananda', 'eightfold', 'right',
    'wrong', 'view', 'resolve', 'speech', 'action', 'livelihood', 'effort', 'mindfulness',
    'samiddhi', 'concentration', 'cessation', 'origination', 'birth', 'death', 'pleasure',
    'pain', 'good', 'bad', 'north', 'south', 'skin', 'flesh', 'arm', 'leg', 'eye', 'ear',
    'smell', 'sound', 'sense', 'king', 'blue', 'five', 'cow', 'bird', 'day', 'queen', 'father', 'mother',
    'sword', 'ax', 'sensation', 'consciousness', 'dukkha', 'jhana', 'jeta', 'dhukka', 'monk',
    'bhikku', 'brahman', 'recluse'
]


def read_data(path):
    """Reads input data."""

    sentences = []

    with open(path, 'r') as infile:
        for line in infile.read().split('\n'):
            words = line.split(' ')
            sentences.append([w for w in words if len(w) > 0])

    return sentences


def plot_points(points, labels, filename=None):
    """Plots a set of x,y points with labels."""

    plt.figure(figsize=(64, 64))

    for i, label in enumerate(labels):
        x, y = points[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    if filename:
        plt.savefig(filename)


def train():
    """Trains the model."""

    sentences = read_data(INPUT_PATH)
    scanner = skipgram.SkipGramScanner(
        sentences, window=WINDOW_SIZE, sample=N_SUBSAMPLING)
    check_ids = [scanner.get_id(w) for w in CHECK_WORDS if w in scanner.vocabulary]
    n_words = len(scanner.vocabulary)

    word_vectors = model.WordVectors(
        vocabulary_size=len(scanner.vocabulary),
        embedding_size=EMBEDDING_SIZE,
        n_sampled=N_SAMPLED,
        learning_rate=LEARNING_RATE)

    saver = tf.train.Saver()

    with tf.Session() as session:

        writer = tf.summary.FileWriter(LOG_PATH, session.graph)
        init = tf.global_variables_initializer()
        init.run()

        average_loss = 0

        for step in range(1, N_STEPS):

            batch, labels = scanner.batch(BATCH_SIZE)
            _, loss, summary = word_vectors.train(session, batch, labels)
            average_loss += loss
            writer.add_summary(summary, step)

            if step % 1000 == 0:
                print('average loss at step {}: {}'.format(step, average_loss / 1000))
                saver.save(session, os.path.join(LOG_PATH, 'model'), step)
                average_loss = 0

            if step % 10000 == 0:
                for n in word_vectors.nearest(session, check_ids):
                    print('{} -> {}'.format(
                        scanner.get_label(n[0]), [scanner.get_label(i) for i in n[1:]]))

        if not os.path.exists(LOG_PATH):
            os.makedirs(LOG_PATH)

        with open(os.path.join(LOG_PATH, 'vocabulary.txt'), 'w') as outfile:
            outfile.write('\n'.join([scanner.get_label(i) for i in range(n_words)]))
            print('wrote vocabulary -> {}'.format(outfile.name))

        outpath = saver.save(session, os.path.join(LOG_PATH, 'model'))
        print('wrote model -> {}'.format(outpath))

        writer.close()

        outpath = os.path.join(LOG_PATH, 'word2vec.bin')
        word2vec.word2vec('data/sentences.txt', outpath, size=EMBEDDING_SIZE)


def evaluate():
    """Evaluates the model."""

    tf.reset_default_graph()

    with open(os.path.join(LOG_PATH, 'vocabulary.txt')) as infile:
        words = infile.read().split('\n')
        reverse_dictionary = {i: w for i, w in enumerate(words)}
        dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))

    word_vectors = model.WordVectors(
        vocabulary_size=len(dictionary),
        embedding_size=EMBEDDING_SIZE,
        n_sampled=N_SAMPLED,
        learning_rate=LEARNING_RATE)

    saver = tf.train.Saver()

    with tf.Session() as session:

        saver.restore(session, os.path.join(LOG_PATH, 'model'))

        check_ids = [dictionary[w] for w in CHECK_WORDS if w in dictionary]
        print('model >>>>>>>>>>>>>>>>>>>>>>>>>')
        for n in word_vectors.nearest(session, check_ids):
            print('{} -> {}'.format(reverse_dictionary[n[0]], ', '.join(
                [reverse_dictionary[i] for i in n[1:]])))

        tsne = TSNE(n_components=2, init='pca', verbose=1)
        labels = [reverse_dictionary[i] for i in range(N_PLOT)]
        points = tsne.fit_transform(word_vectors.embeddings(session)[:N_PLOT, :])
        plot_points(points, labels, filename='tsne.png')

    w2v = word2vec.load('data/model/word2vec.bin')

    print('word2vec >>>>>>>>>>>>>>>>>>>>>>>>>')
    for word in CHECK_WORDS:
        if word in w2v.vocab:
            indexes, metrics = w2v.cosine(word)
            nearest = [c[0] for c in w2v.generate_response(indexes, metrics)]
            print('{} -> {}'.format(word, nearest))


if __name__ == '__main__':

    if not os.path.isfile(os.path.join(LOG_PATH, 'model.index')):
        train()

    evaluate()
