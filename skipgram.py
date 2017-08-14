import collections
import numpy as np
import math


class SkipGramScanner(object):
    """Scans over a list of sentences to produce id-encoded skip grams."""

    def __init__(self, sentences, window=5, sample=1e3, threshold=5):

        self.window = window
        self.threshold = threshold
        self.sample = sample
        self.ngrams = []
        self.index = 0
        self.counter = collections.Counter()
        self.total_words = 0
        self.vocabulary = set()

        # count all words
        for sentence in sentences:
            self.counter.update(sentence)

        # filter out words < threshold from vocabulary
        top_words = [w for w in self.counter.most_common() if w[1] >= self.threshold]
        self.counter = collections.Counter(dict(top_words))

        # ids in descending order by frequency
        self.dictionary = {c[0]: i for i, c in enumerate(self.counter.most_common())}
        self.reverse_dictionary = dict(
            zip(self.dictionary.values(), self.dictionary.keys()))

        encoded = []

        # encode word -> id, omit words < threshold
        for sentence in sentences:
            selected = [w for w in sentence if w in self.counter]
            encoded.append([self.dictionary[w] for w in selected])
            self.vocabulary.update(selected)
            self.total_words += len(selected)

        self.sentences = encoded

    def get_id(self, word):
        "Returns the id for a word." ""

        return self.dictionary.get(word)

    def get_label(self, id_):
        """Returns the word for an id."""

        return self.reverse_dictionary.get(id_)

    def get_skipgrams(self, sentence):
        """Extracts skip grams from a sentence."""

        selected = []

        for word_id in sentence:

            word = self.reverse_dictionary[word_id]

            # subsample frequent words
            frequency = self.counter[word] / self.total_words
            sample_prob = (math.sqrt(frequency / self.sample) + 1) * (
                self.sample / frequency)

            if np.random.sample() > sample_prob:
                continue

            selected.append(word_id)

        sentence = selected
        skipgrams = []

        for i, word_id in enumerate(selected):

            # sample context window between 1 and window size
            window = np.random.randint(low=1, high=self.window + 1)
            start = max(i - window, 0)
            end = min(i + window, len(selected) - 1)

            # add all context words
            for j in range(start, end + 1):
                if j == i:
                    continue

                skipgrams.append((word_id, selected[j]))

        return skipgrams

    def batch(self, size):
        """Returns a batch of skip grams."""

        batch = []
        labels = []

        # fill a batch of ngrams
        while len(batch) < size:
            counter = 0

            # extract from sentences until buffer has ngrams
            while not self.ngrams:
                self.ngrams = self.get_skipgrams(self.sentences[self.index])
                self.index = (self.index + 1) % len(self.sentences)
                counter += 1

                # just in case
                if counter > 100:
                    raise Exception('failed to extract ngrams')

            ngram = self.ngrams.pop(0)
            batch.append(ngram[0])
            labels.append(ngram[1])

        batch = np.array(batch)
        labels = np.array([[l] for l in labels])

        return batch, labels
