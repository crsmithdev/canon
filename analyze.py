# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Basic word2vec example."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import collections
import glob
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
import nltk

with open('data/source/stopwords/stopwords.txt', 'r') as infile:
    stopwords = infile.read().strip().split('\n')

def convert_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def read_data():

    data = []
    path = 'data/processed/*.txt'
    wn = nltk.stem.WordNetLemmatizer()

    for infile_path in glob.iglob(path):
        with open(infile_path, 'r') as infile:

            text = re.sub(r'\n', ' ', infile.read())
            text = text.lower()
            text = re.sub(r'[.,?!—\-\":;\*\(\)“”]', ' ', text)
            text = re.sub(r"&", "and", text)
            text = re.sub(r"\[[^\]]*\]", "", text)
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r'\'s', ' s', text)
            text = re.sub(r'doesn\'t', 'does not', text)
            text = re.sub(r'don\'t', 'do not', text)
            text = re.sub(r'didn\'t', 'did not', text)
            text = re.sub(r'isn\'t', 'is not', text)
            text = re.sub(r'wouldn\'t', 'would not', text)
            text = re.sub(r'haven\'t', 'have not', text)
            text = re.sub(r'can\'t', 'can not', text)
            text = re.sub(r'won\'t', 'will not', text)
            text = re.sub(r'hasn\'t', 'has not', text)
            text = re.sub(r'shouldn\'t', 'should not', text)
            text = re.sub(r'couldn\'t', 'could not', text)
            text = re.sub(r'wasn\'t', 'was not', text)
            text = re.sub(r'weren\'t', 'were not', text)
            text = re.sub(r'aren\'t', 'are not', text)
            text = re.sub(r'i\'ll', 'i will', text)
            text = re.sub(r'we\'ll', 'we will', text)
            text = re.sub(r'i\'m', 'i am', text)
            text = re.sub(r'we\'re', 'we are', text)
            text = re.sub(r'you\'re', 'you are', text)
            text = re.sub(r'they\'re', 'they are', text)
            text = re.sub(r'i\'ve', 'i have', text)
            text = re.sub(r'0', 'zero', text)
            text = re.sub(r'1', 'one', text)
            text = re.sub(r'2', 'two', text)
            text = re.sub(r'3', 'three', text)
            text = re.sub(r'4', 'four', text)
            text = re.sub(r'5', 'five', text)
            text = re.sub(r'6', 'six', text)
            text = re.sub(r'7', 'seven', text)
            text = re.sub(r'8', 'eight', text)
            text = re.sub(r'9', 'nine', text)
            text = re.sub(r' ii ', 'two', text)
            text = re.sub(r' iii ', 'three', text)
            text = re.sub(r' iv ', 'four', text)
            text = re.sub(r' v ', 'five', text)
            text = re.sub(r' vi ', 'six', text)
            text = re.sub(r' vii ', 'seven', text)
            text = re.sub(r' viii ', 'eight', text)
            text = re.sub(r' ix ', 'nine', text)
            text = re.sub(r'ā', 'a', text)
            text = re.sub(r'ṇ', 'n', text)
            text = re.sub(r'ḍ', 'd', text)
            text = re.sub(r'ñ', 'n', text)
            text = re.sub(r'ṅ', 'n', text)
            text = re.sub(r'ū', 'u', text)
            text = re.sub(r'ī', 'i', text)
            text = re.sub(r'ṭ', 't', text)
            for c in text:
                if ord(c) > 128:
                    print(c)


            text = text.lower()

            words = text.split(' ')
            if "don't" in words:
                print("*****")
            words = [w.strip('\'') for w in words]
            words = [w for w in words if len(w) > 0]
            words = [w for w in words if len(w) > 1 or w in {'a', 'i'}]

            tagged = nltk.pos_tag(words)
            tagged = [(t[0], convert_tag(t[1])) for t in tagged]
            words = [wn.lemmatize(t[0], pos=t[1]) for t in tagged]
            #words = [re.sub(r"\'s$", "", w) for w in words]
            #words = [w for w in words if len(w) > 1]

            #data.extend(deplural)
            data.extend(words)
            if "don't" in words:
                print("!!!!")

            outpath = 'data/temp/' + os.path.basename(infile_path)
            with open(outpath, 'w+') as outfile:
                outfile.write(' '.join(words))
                print('{} -> {}'.format(infile_path, outpath))

    return data

vocabulary = read_data()
print('# total words: {}'.format(len(vocabulary)))
print('# unique words: {}'.format(len(set(vocabulary))))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 3000


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                            vocabulary_size)
del vocabulary  # Hint to reduce memory.

with open('data/temp/counts.txt', 'w') as outfile:
    lines = ['{} -> {}'.format(w, c) for w, c in count]
    text = '\n'.join(lines)
    outfile.write(text)

#print(len([c for c in count if c[1] > 5]))
#print('Most common words (+UNK)', count[:5])
#print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

#print(count[:vocabulary_size])
data_index = 0


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
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]],
          '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 100     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
valid_examples = [i for i in range(vocabulary_size)]
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(weights=nce_weights,
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
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

# Step 5: Begin training.
num_steps = 50001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
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
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    if nearest[k] not in reverse_dictionary:
                        print(nearest[k], k)
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = '%s %s,' % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()

    save_path = saver.save(session, "data/temp/model.ckpt")
    print("Model saved in file: %s" % save_path)

# Step 6: Visualize the embeddings.


def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(36, 36))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE, Isomap
    import matplotlib.pyplot as plt

    isomap = Isomap(10, n_components=2)
    tsne = TSNE(perplexity=50, n_components=2, init='pca', n_iter=5000, early_exaggeration=8.0, verbose=1)
    plot_only = 1500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    #low_dim_embs2 = isomap.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)

except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')

#for k in xrange(top_k):
#                    if nearest[k] not in reverse_dictionary:
#                        print(nearest[k], k)
#                    close_word = reverse_dictionary[nearest[k]]
#                    log_str = '%s %s,' % (log_str, close_word)
#                print(log_str)
