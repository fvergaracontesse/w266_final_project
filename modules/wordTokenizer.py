#tokenize titles

"""Word tokenizer class"""

import os
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class WordTokenizer(object):

    def __init__(self, max_sequence_length=200, prefix="./models/word_tokenizer"):
        self.max_sequence_length = max_sequence_length
        self.prefix = prefix
        self.tokenizer = None

    def save(self, prefix=None):
        if prefix != None: self.prefix = prefix
        pickle.dump(self.tokenizer, open(self.prefix+".pickle", "wb"))

    def load(self, prefix=None):
        if prefix != None: self.prefix = prefix
        self.tokenizer = pickle.load(open(self.prefix+".pickle", "rb"))

    def train(self, texts, max_nb_words=80000):
        # Tokenize
        print('Training tokenizer...')
        self.tokenizer = Tokenizer(num_words=max_nb_words)
        self.tokenizer.fit_on_texts(texts)
        self.save()
        print(('Found %s unique tokens.' % len(self.tokenizer.word_index)))

    def tokenize(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        data = pad_sequences(sequences, maxlen=self.max_sequence_length)
        return data
