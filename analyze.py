from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import collections
import glob
import math
import os
import random
import pickle
import zipfile

import numpy as np
import tensorflow as tf

INPUT_PATH = 'data/processed/*.txt'

TEMP_PATH = 'data/temp'

STOPWORDS_PATH = 'data/source/stopwords/stopwords.txt'

with open(STOPWORDS_PATH, 'r') as file:
    stopwords = file.read().strip().split('\n')

class NGramScanner(object):

    def __init__(self, sentences, window, reverse_dictionary):
        self.sentences = sentences
        self.window = window
        self.ngrams = []
        self.index = 0

        self.reverse_dictionary = reverse_dictionary

    def _extract(self, sentence):

        ngrams = []
        #print([reverse_dictionary.get(i, i) for i in sentence])

        for i, word in enumerate(sentence):
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

        #for i in range(len(batch)):
        #    print(batch[i], reverse_dictionary[batch[i]],
        #        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

        return batch, labels


def read_data():

    with open(STOPWORDS_PATH, 'r') as file:
        stopwords = file.read().strip().split('\n')

    words = []
    sentences = []

    for path in glob.iglob(INPUT_PATH):
        with open(path, 'r') as file:

            text = file.read()
            lines = text.split('\n')

            for line in lines:
                sentence_words = []
                line_words = line.split(' ')
                for w in line_words:
                    w = re.sub(r'("|\')', '', w)
                    words.extend(w.split('-'))
                    sentence_words.append(w)

                sentence_words = [w for w in sentence_words if re.match(r'[a-z]+', w)]
                sentence_words = [w for w in sentence_words if len(w) > 1 or w in {'a', 'i'}]
                #print(sentence_words)

                if len(sentence_words) > 0:
                    sentences.append(sentence_words)


    words = [w for w in words if re.match(r'[a-z]+', w)]
    words = [w for w in words if len(w) > 1 or w in {'a', 'i'}]
    #words = [w for w in words if w not in stopwords]
    #for s in sentences[0:20]:
    #    print(s)

    return words, sentences


def plot_points(points, labels, filename=None):

    plt.figure(figsize=(64, 64))  # in inches

    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    if filename:
        plt.savefig(filename)


data_index = 0
data_index2 = 0

def build_dataset2(words, n_words):

    counter = collections.Counter(words)

    counts = [['UNK', -1]]
    counts.extend(counter.most_common(n_words - 1))

    dictionary = {c[0]: i for i, c in enumerate(counts)}
    data = []
    unk_count = 0

    for word in words:
        index = dictionary.get(word, 0)

        if index:
            unk_count += 1

        data.append(index)

    counts[0][1] = unk_count

    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, counts, dictionary, reversed_dictionary

def generate_batch2(batch_size, num_skips, skip_window):
    global data_index2

    batch = []
    labels = []

    while len(batch) < batch_size:
        start = max(data_index2 - (2 * skip_window) - 1, 0)
        end = min(data_index2 + (2 * skip_window) + 1, len(data) - 1)
        span = data[start:end]
        for i, target in enumerate(span):
            #if target in stopword_ids:
            if target == 0:
                continue
            target_start = max(i - skip_window, 0)
            target_end = min(i + skip_window, len(span) - 1)
            #print(reverse_dictionary[target])
            for j in range(target_start, target_end + 1):
                if j != i and span[j] != 0:
                    #if j != i and span[j] not in stopword_ids:
                    #print(' -> ' + reverse_dictionary[span[j]])
                    batch.append(target)
                    labels.append(span[j])
                if len(batch) == batch_size:
                    break

            if len(batch) == batch_size:
                break

        data_index2 = (data_index2 + len(span)) % len(data)

    #for i in range(len(batch)):
    #    print(batch[i], reverse_dictionary[batch[i]],
    #        '->', labels[i], reverse_dictionary[labels[i]])

    #print(len(batch))
    #print(data_index2)

    batch = np.array(batch)
    labels = np.array([[l] for l in labels])

    return batch, labels

    #print(i, reverse_dictionary[target], target_start, target_end)


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    #print([reverse_dictionary[i] for i in buffer])
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

        #print([reverse_dictionary[i] for i in buffer])

    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)

    #for i in range(len(batch)):
    #    print(batch[i], reverse_dictionary[batch[i]],
    #        '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
    #print(data_index)

    #print(len(batch), len(labels))
    nz_b = len([b for b in batch if b > 0])
    nz_l = len([l for l in labels if l > 0])
    #if nz_b != nz_l != batch_size:
    #    print(nz_b, nz_l)

    #print(len([b for b in batch if b > 0]), len([l for l in labels if l > 0]))
    #print(labels[0])

    return batch, labels



# Step 4: Build and train a skip-gram model.
#vocabulary = read_data()
saved = os.path.isdir(TEMP_PATH)  #  .isfile('data/temp/model.ckpt.index')
print('using saved: {}'.format(saved))

vocabulary_size = 3000

#if saved:
#    with open('data/temp/dictionary.pickle', 'rb') as infile:
#        reverse_dictionary = pickle.load(infile)
#if not saved:
vocabulary, sentences = read_data()
print('# total words: {}'.format(len(vocabulary)))
print('# unique words: {}'.format(len(set(vocabulary))))

# Step 2: Build the dictionary and replace rare words with UNK token.

data, count, dictionary, reverse_dictionary = build_dataset2(vocabulary, vocabulary_size)
stopword_ids = [dictionary.get(k) for k in stopwords]
stopword_ids = [i for i in stopword_ids if i is not None]
stopword_ids.append(0)
data = [d for d in data if d != 0 and d not in stopword_ids]
#for i in range(len(sentences)):
for i in range(len(sentences)):
    source = sentences[i]
    ids = [dictionary.get(w, 0) for w in source]
    filtered = [i for i in ids if i != 0 and i not in stopword_ids]
    print(sentences[i])
    sentences[i] = filtered
    print([reverse_dictionary[i] for i in filtered])
print(count[len(count) - 1])
print(len([c for c in count if c[1] > 10]))
#for s in sentences[:20]:
#    print(s)
#for c in count:
#    print('{} -> {}'.format(c[0], c[1]))
del vocabulary  # Hint to reduce memory.

#with open('data/temp/counts.txt', 'w') as outfile:
#    lines = ['{} -> {}'.format(w, c) for w, c in count]
#    text = '\n'.join(lines)
#    outfile.write(text)

scanner = NGramScanner(sentences, 8, reverse_dictionary)
batch, labels = scanner.batch(300)
#print(batch, labels)

#exit()
if not os.path.exists('data/temp'):
    os.makedirs('data/temp')

with open('data/temp/dictionary.pickle', 'wb') as outfile:
    pickle.dump(reverse_dictionary, outfile)


#generate_batch(128, 4, 4)
#generate_batch2(128, 4, 4)
#generate_batch(128, 4, 4)
#generate_batch2(128, 4, 4)
#exit()

batch_size = 256
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 5  # How many words to consider left and right.
num_skips = 4  # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 100  # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
valid_examples = [i for i in range(vocabulary_size)]
num_sampled = 12  # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=train_labels,
            inputs=embed,
            num_sampled=num_sampled,
            num_classes=vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

# Step 5: Begin training.
num_steps = 200001
check_words = [
    'monk', 'blessed', 'mind', 'body', 'dhamma', 'right', 'wrong', 'cessation', 'origination', 'feeling', 'form',
    'perception', 'birth', 'death', 'pleasure', 'pain', 'noble', 'consciousness', 'brahman', 'good', 'bad', 'mental',
    'bodily', 'four', 'six', 'eight', 'tathagata', 'king', 'master', 'north', 'first', 'skin', 'flesh', 'ananda', 'ear',
    'eye', 'nose', 'tongue', 'two', 'three', 'gotama'
]

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0

    if saved:
        saver.restore(session, 'data/temp/model.ckpt')
    else:
        for step in range(num_steps):
            #batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            #batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            batch_inputs, batch_labels = scanner.batch(batch_size)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

            # Note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
                sim = similarity.eval()
                for w in check_words:
                    valid_word = w
                    i = dictionary[w]
                    #for i in range(valid_size):
                    #    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8  # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = 'Nearest to %s:' % valid_word
                    for k in range(top_k):
                        if nearest[k] not in reverse_dictionary:
                            print(nearest[k], k)
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = '%s %s,' % (log_str, close_word)
                    print(log_str)
                    #print(data_index2)

        save_path = saver.save(session, "data/temp/model.ckpt")
        print("Model saved in file: %s" % save_path)

    final_embeddings = normalized_embeddings.eval()
    sim = similarity.eval()

# Step 6: Visualize the embeddings.


def filter_inclusive(embeddings, labels, ids):

    embeddings = np.array([e for i, e in enumerate(embeddings) if i in ids])
    labels = np.array([l for i, l in enumerate(labels) if i in ids])

    return embeddings, labels


def filter_exclusive(embeddings, labels, ids):

    embeddings = np.array([e for i, e in enumerate(embeddings) if not i in ids])
    labels = np.array([l for i, l in enumerate(labels) if i not in ids])

    return embeddings, labels


def closest(id, similarity, n):
    nearest = (-similarity[i, :]).argsort()[0:n + 1]
    return nearest


try:
    with open(STOPWORDS_PATH, 'r') as file:
        stopwords = file.read().strip().split('\n')
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE, Isomap, MDS, LocallyLinearEmbedding
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    isomap = Isomap(10, n_components=2)
    #tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=2000, early_exaggeration=8.0, angle=0.15, verbose=1)
    tsne = TSNE(n_components=2, init='pca', n_iter=2000, verbose=1)
    lle = LocallyLinearEmbedding(method='modified')
    mds = MDS(n_components=2)

    plot_only = 1500
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    dictionary = dict(zip(reverse_dictionary.values(), reverse_dictionary.keys()))

    stopword_ids = [dictionary.get(k) for k in stopwords]
    stopword_ids = [i for i in stopword_ids if i is not None]

    #labels = ['two', 'consciousness', 'form', 'perception', 'cessation', 'dhamma']

    words = ['eye', 'ear', 'consciousness']
    word_ids = [dictionary[w] for w in words]
    for i in range(500):
        if i not in stopword_ids:
            label = reverse_dictionary[i]
            ids = closest(i, sim, 10)

            print('{} -> {}'.format(label, ', '.join([reverse_dictionary[i] for i in ids[1:]])))

    e, l = filter_exclusive(final_embeddings, labels, stopword_ids)
    clusters = 25
    kmeans_clustering = KMeans(n_clusters=clusters)

    low_dim_embs = tsne.fit_transform(e[:plot_only, :])
    plot_points(low_dim_embs, l, filename='tsne.png')
    print(low_dim_embs.shape)

    #print(cluster_words)
    idx = kmeans_clustering.fit_predict(e[:plot_only, :])
    print(idx.shape)
    cluster_words = [[] for i in range(clusters)]
    for i in range(len(idx)):
        cluster = idx[i]
        print(cluster, i)
        word = reverse_dictionary[i]
        #word = i
        cluster_words[cluster].append(word)

    for i in range(clusters):
        print('cluster {}'.format(i))
        print(', '.join(cluster_words[i]))

    #low_dim_embs = isomap.fit_transform(final_embeddings[:plot_only, :])

    #low_dim_embs = mds.fit_transform(final_embeddings[:plot_only, :])

except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')

#for k in xrange(top_k):
#                    if nearest[k] not in reverse_dictionary:
#                        print(nearest[k], k)
#                    close_word = reverse_dictionary[nearest[k]]
#                    log_str = '%s %s,' % (log_str, close_word)
#                print(log_str)
