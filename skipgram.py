import collections
import numpy as np
import math


class SkipGramScanner(object):

    def __init__(self, sentences, window=5, sample=1e3):

        self.window = window
        self.ngrams = []
        self.index = 0
        self.sample = sample
        self.counter = collections.Counter()
        self.total_words = 0

        for sentence in sentences:
            self.counter.update(sentence)
            self.total_words += len(sentence)

        self.dictionary = {c[0]: i for i, c in enumerate(self.counter.most_common())}
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))

        encoded = []

        for sentence in sentences:
            encoded.append([self.dictionary[w] for w in sentence])

        self.sentences = encoded

    def _extract(self, sentence):

        ngrams = []
        selected = []

        for word_id in sentence:

            word = self.reverse_dictionary[word_id]

            if self.counter[word] < 5:
                continue

            frequency = self.counter[word] / self.total_words
            sample_prob = (math.sqrt(frequency / self.sample) + 1) * (self.sample / frequency)

            if np.random.sample() > sample_prob:
                continue

            selected.append(word_id)

        sentence = selected

        for i, word_id in enumerate(sentence):

            window = np.random.randint(low=1, high=self.window + 1)
            start = max(i - window, 0)
            end = min(i + window, len(sentence) - 1)

            for j in range(start, end + 1):
                if j == i:
                    continue

                ngrams.append((word_id, sentence[j]))

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
