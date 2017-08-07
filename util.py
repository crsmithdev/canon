import glob
import re
import matplotlib.pyplot as plt
import collections


def read_data(inpath):

    sentences = []

    for path in glob.iglob(inpath):
        with open(path, 'r') as infile:

            lines = infile.read().split('\n')
            sentences.extend([l.split(' ') for l in lines])

    return sentences

def encode_data(sentences, threshold=5, ignore=None):

    ignore = ignore if ignore else set()
    counter = collections.Counter()
    total_words = 0
    sentences_out = []

    for sentence in sentences:
        counter.update(sentence)
        total_words += len(sentence)

    print('total:', total_words)

    counts = counter.most_common()
    dictionary = {c[0]: i for i, c in enumerate(counts)}
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    for sentence in sentences:
        encoded = []
        for word in sentence:
            encoded.append(dictionary[word])

        sentences_out.append(encoded)

    return sentences_out, counter, total_words, dictionary, reverse_dictionary


def plot_points(points, labels, filename=None):

    plt.figure(figsize=(64, 64))  # in inches

    for i, label in enumerate(labels):
        x, y = points[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    if filename:
        plt.savefig(filename)
