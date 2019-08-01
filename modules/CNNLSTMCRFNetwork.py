import os, json
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed, Activation, concatenate, Conv1D, GlobalMaxPooling1D,Flatten, Input, Dropout
from keras.models import load_model, Sequential, Model
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy
from keras_contrib.metrics import crf_marginal_accuracy
from keras_contrib.metrics import crf_viterbi_accuracy

from sklearn.metrics import classification_report

class CNNLSTMCRFNetwork(object):

    def __init__(self, prefix=None):

        if prefix != None:
            self.load(prefix)
        else:
            self.prefix = 'models/CNNlstmCRF'
            self.model = None
            self.modelCNN = None
            self.concatenate = None
            self.tag_map = {}

    def load(self, prefix=None):

        if prefix != None: self.prefix = prefix
        self.model = load_model(self.prefix+'.h5',
               custom_objects={'CRF': CRF,
                               'crf_loss': crf_loss,
                               'crf_viterbi_accuracy': crf_viterbi_accuracy})
        self.tag_map = json.load(open(self.prefix+'.json', 'r'))

    def save(self, prefix=None):
        if prefix != None: self.prefix = prefix
        self.model.save(self.prefix+'.h5')
        with open(self.prefix+'.json', 'w') as out:
            json.dump(self.tag_map, out)

    def tag(self, data):
        prediction = self.model.predict(data)
        all_tag_probs = []
        dataAdj = data[0]
        for i in range(prediction.shape[0]):
            sentence_tag_probs = []
            first_word = 0
            for j in range(dataAdj[i].shape[0]):
                if dataAdj[i,j] != 0: break
                first_word += 1
            for j in range(first_word, prediction.shape[1]):
                word_tag_probs = {}
                for tag in self.tag_map:
                    word_tag_probs[tag] = prediction[i,j,self.tag_map[tag]-1]
                sentence_tag_probs.append(word_tag_probs)
            all_tag_probs.append(sentence_tag_probs)
        return all_tag_probs

    def index_tags(self, tags):
        indices = []
#        self.tag_map = json.load(open(self.prefix+'.json', 'r'))
        for tag in tags:
            if not (tag in self.tag_map):
                self.tag_map[tag] = len(self.tag_map) + 1
            indices.append(self.tag_map[tag])
        return indices

    def get_labels(self, tag_sets, tokenizer):
        labels = []
        print('Getting labels...')
        for tag_set in tag_sets:
            indexed_tags = self.index_tags(tag_set)
            labels.append(to_categorical(np.asarray(indexed_tags), num_classes=4)[:,1:])
        labels = pad_sequences(labels, maxlen=tokenizer.max_sequence_length)
        return labels

    def compile(self, wordTokenizer,charTokenizer, data_dir='./data/', embedding_dim=300, dropout_fraction=0.5, hidden_dim=32, embedding_file='glove-sbwc.i25.vec'):

        # Load embedding layer
        print('Loading GloVe embedding...')
        embeddings_index = {}
        f = open(os.path.join(data_dir, embedding_file), 'r')
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        print(('Found %s word vectors.' % len(embeddings_index)))

        # Create embedding layer
        print('Creating embedding layer...')
        embedding_matrix = np.zeros((len(wordTokenizer.tokenizer.word_index) + 1, embedding_dim))
        for word, i in list(wordTokenizer.tokenizer.word_index.items()):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector

        # Create network
        print('Creating network...')

        inputWordEmbeddings = Input(shape=(wordTokenizer.max_sequence_length,),name="words_input")
        inputCharEmbeddings = Input(shape=(wordTokenizer.max_sequence_length,charTokenizer.max_sequence_length,),name="char_input")

        print('Start word embeddings')

        wordsEmbeddings = Embedding(len(wordTokenizer.tokenizer.word_index) + 1,
                                 embedding_dim,
                                 weights=[embedding_matrix],
                                 input_length=wordTokenizer.max_sequence_length,
                                 trainable=False,
                                 mask_zero=True)(inputWordEmbeddings)

        print('Start char embeddings')
        charEmbeddings = TimeDistributed(Embedding(len(charTokenizer.tokenizer.word_index) + 1, output_dim=10,
                           input_length=charTokenizer.max_sequence_length))(inputCharEmbeddings)

        dropout= Dropout(dropout_fraction)(charEmbeddings)

        conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=10, padding='same',activation='tanh', strides=1))(dropout)

        maxpool_out=TimeDistributed(GlobalMaxPooling1D())(conv1d_out)

        char = TimeDistributed(Flatten())(maxpool_out)

        char = Dropout(dropout_fraction)(char)

        print('Concat models')

        output = concatenate([wordsEmbeddings, char])

        output = Bidirectional(LSTM(hidden_dim, return_sequences=True))(output)

        output = TimeDistributed(Dense(hidden_dim, activation='relu'))(output)


        print('Add crf layer')

        crf = CRF(len(self.tag_map))

        output = crf(output)

        self.model = Model(inputs=[inputWordEmbeddings, inputCharEmbeddings], outputs=[output])

        # Compile model
        print('Compiling network...')
        self.model.compile(loss=crf.loss_function,
                           optimizer='adam',
                           metrics=[crf.accuracy])

    def train(self, data, labels, validation_split=0.2, batch_size=256, epochs=3):

        print('Training...')
        # Split the data into a training set and a validation set
        np.random.seed(seed=1)
        indices = np.arange(data[0].shape[0])
        np.random.shuffle(indices)
        dataWords = data[0][indices]
        dataChars = data[1][indices]
        labels = labels[indices]
        nb_validation_samples = int(validation_split * dataWords.shape[0])

        x_train_words = dataWords[:-nb_validation_samples]
        x_train_char  = dataChars[:-nb_validation_samples]
        y_train = labels[:-nb_validation_samples]
        x_val_words = dataWords[-nb_validation_samples:]
        x_val_char  = dataChars[-nb_validation_samples:]
        y_val = labels[-nb_validation_samples:]

        print(dataWords.shape,dataChars, labels.shape)

        # Train!
        self.save()
        checkpointer = ModelCheckpoint(filepath=self.prefix+'.h5', verbose=1, save_best_only=False)
        self.model.fit([x_train_words,x_train_char], y_train, validation_data=([x_val_words,x_val_char], y_val),
                       callbacks=[checkpointer],
                       epochs=epochs, batch_size=batch_size)
        self.model.summary
        self.evaluate([x_val_words,x_val_char], y_val, batch_size)

    def evaluate(self, x_test, y_test, batch_size=256):

        print('Evaluating...')
        predictions_last_epoch = self.model.predict(x_test, batch_size=batch_size, verbose=1)
        predicted_classes = np.argmax(predictions_last_epoch, axis=2).flatten()
        y_val = np.argmax(y_test, axis=2).flatten()
        target_names = ['']*(max(self.tag_map.values()))
        for category in self.tag_map:
            target_names[self.tag_map[category]-1] = category

        print((classification_report(y_val, predicted_classes, target_names=target_names, digits = 6, labels=range(len(target_names)))))
