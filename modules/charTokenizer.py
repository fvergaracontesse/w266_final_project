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

    def __init__(self, max_sequence_length=10, prefix="./models/charTokenizer",max_sequence_sentence_words=200):
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
        # Tokenize
        print('Training tokenizer...')
        self.tokenizer = Tokenizer(num_words=None,char_level=self.char_level,oov_token=self.oov_token)
        self.tokenizer.fit_on_texts(texts)
        self.save()
        print(self.tokenizer.word_index)


    def tokenize(self, texts):
        #print("texts",texts,len(texts))
        if len(texts)>1:
            data = np.zeros((len(texts),self.max_sequence_sentence_words, self.max_sequence_length))
            for i in range(len(texts)-1):
                words = text_to_word_sequence(texts[i], filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
                #print("entre",words[i])
                sequences = self.tokenizer.texts_to_sequences(words)
                #print("sequence",sequences)
                pad = pad_sequences(sequences, maxlen=self.max_sequence_length,padding='post')
            #print(pad,pad.shape)
                for j in range(len(words)-1):
                    for k in range(9):
                        data[i][j][k] = pad[j][k]
                #sequences = self.tokenizer.texts_to_sequences(words[j])
                #pad = pad_sequences(sequences.reshape(), maxlen=self.max_sequence_length)
                #print(sequences)
            #print(words)
            #if len(words)>200:
            #    sequences = list(map(lambda x: self.tokenizer.texts_to_sequences(x),words[0:200]))
            #else:
            #    sequences = list(map(lambda x: self.tokenizer.texts_to_sequences(x),words))
            #print(sequences)
            #pad_sequence = pad_sequences(sequences, maxlen=self.max_sequence_length)
            #print(pad_sequence,type(pad_sequence))
            #seq_list = pad_sequence.reshape(len(words),self.max_sequence_length)
            #data.append([seq_list])
        else:
            i=0
            data = np.zeros((len(texts),200, self.max_sequence_length))
            words = text_to_word_sequence(texts[i], filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ')
            sequences = self.tokenizer.texts_to_sequences(words)
            pad = pad_sequences(sequences, maxlen=self.max_sequence_length,padding='post')
            for j in range(len(words)-1):
                for k in range(9):
                    data[i][j][k] = pad[j][k]
        return data

