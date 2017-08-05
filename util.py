import glob
import re
import matplotlib.pyplot as plt
import collections


def read_data(inpath):

    sentences = []
    vocabulary = set()

    for path in glob.iglob(inpath):
        with open(path, 'r') as infile:
            lines = infile.read().split('\n')

            for line in lines:

                words = line.split(' ')
                words = [w for w in words if re.match(r'^[a-z/-]+$', w)]  # only letters, - and /
                words = [w for w in words if len(w) > 1 or w in {'a', 'i'}]  # only certain single-letter words

                # split words with dashes into two words
                split = [w.split('-') for w in words]
                words = [w for s in split for w in s]

                vocabulary.update(words)

                # skip if empty after filtering
                if words:
                    sentences.append(words)

    # fix straggling plural words with whole vocabulary in hand
    plurals = {p: s for s, p in [(w, w + 's') for w in vocabulary] if p in vocabulary}
    for i, sentence in enumerate(sentences):
        sentences[i] = [plurals[w] if w in plurals else w for w in sentence]

    return sentences


def encode_data(sentences, threshold=10, ignore=None):

    ignore = ignore if ignore else set()
    counter = collections.Counter()
    sentences_out = []

    for sentence in sentences:
        counter.update(sentence)

    counts = counter.most_common()
    counts = [c for c in counts if c[1] >= threshold and c[0] not in ignore]
    dictionary = {c[0]: i for i, c in enumerate(counts)}
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    for k in dictionary.keys():
        plural = dictionary.get(k + 's')
        if plural:
            print(k, reverse_dictionary[plural])

    for sentence in sentences:
        encoded = [dictionary[w] for w in sentence if w in dictionary]
        sentences_out.append(encoded)

    return sentences_out, dictionary, reverse_dictionary


def plot_points(points, labels, filename=None):

    plt.figure(figsize=(64, 64))  # in inches

    for i, label in enumerate(labels):
        x, y = points[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')

    if filename:
        plt.savefig(filename)
