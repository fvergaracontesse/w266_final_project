#tokenize titles

"""Word tokenizer class"""

import os
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

class CharTokenizer(object):

    def __init__(self, max_sequence_length=10, prefix="./models/char_tokenizer",max_sequence_sentence_words=200):
        self.prefix = prefix
        self.max_sequence_length = max_sequence_length
        self.char_level = True
        self.oov_token='UNK'
        self.max_sequence_sentence_words = max_sequence_sentence_words

    def load(self, prefix=None):
        """Loads the tokenizer
        """
        if prefix != None: self.prefix = prefix
        self.tokenizer = pickle.load(open(self.prefix+".pickle", "rb"))

    def save(self, prefix=None):
        if prefix != None: self.prefix = prefix
        pickle.dump(self.tokenizer, open(self.prefix+".pickle", "wb"))

    def train(self, texts):
        print('Training tokenizer...')
        self.tokenizer = Tokenizer(num_words=None,char_level=self.char_level,oov_token=self.oov_token)
        self.tokenizer.fit_on_texts(texts)
        self.save()
        print(self.tokenizer.word_index)


    def tokenize(self, texts):
        if len(texts)>1:
            data = np.zeros((len(texts),self.max_sequence_sentence_words, self.max_sequence_length))
            for i in range(len(texts)-1):
                words = text_to_word_sequence(texts[i], filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
                sequences = self.tokenizer.texts_to_sequences(words)
                pad = pad_sequences(sequences, maxlen=self.max_sequence_length,padding='post')
                for j in range(len(words)-1):
                    for k in range(9):
                        data[i][j][k] = pad[j][k]
        else:
            i=0
            data = np.zeros((len(texts),self.max_sequence_sentence_words, self.max_sequence_length))
            words = text_to_word_sequence(texts[i], filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
            sequences = self.tokenizer.texts_to_sequences(words)
            pad = pad_sequences(sequences, maxlen=self.max_sequence_length,padding='post')
            for j in range(len(words)-1):
                for k in range(9):
                    data[i][j][k] = pad[j][k]
        return data
