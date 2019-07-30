import json
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
from time import time
import io
import re
from csv import reader
#import matplotlib.pyplot as plt

#import matplotlib.pyplot as plt
#from matplotlib import colors
#from matplotlib.ticker import PercentFormatter

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.python.keras.layers import Lambda
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.backend import sparse_categorical_crossentropy
from tensorflow.keras.layers import Dense, TimeDistributed

from sklearn.model_selection import train_test_split

from datetime import datetime

from modules.bertLayer import BertLayer

local_bert_path =   'bert' # change as needed
data_path = 'data/train_products.csv'  # path to ner_dataset.csv file , from

now = datetime.now() # current date and time

# make sure that the paths are accessible within the notebook
sys.path.insert(0,local_bert_path)
sys.path.insert(0,data_path)

import optimization
import run_classifier
import tokenization
import run_classifier_with_tfhub

# Tensorflow hub path to BERT module of choice
bert_url = "https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"

# Define maximal length of input 'sentences' (post tokenization).
max_length = 50

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(bert_url)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])

    return tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()


def addWord(word, ner):
    """
    Convert a word into a word token and add supplied NER and POS labels. Note that the word can be
    tokenized to two or more tokens. Correspondingly, we add - for now - custom 'X' tokens to the labels in order to
    maintain the 1:1 mappings between word tokens and labels.

    arguments: word, pos label, ner label
    """

    # the dataset contains various '"""' combinations which we choose to truncate to '"', etc.
    if word == '""""':
        word = '"'
    elif word == '``':
        word = '`'

    tokens = tokenizer.tokenize(word)
    tokenLength = len(tokens)      # find number of tokens corresponfing to word to later add 'X' tokens to labels

    addDict = dict()

    addDict['wordToken'] = tokens
    #addDict['posToken'] = [pos] + ['posX'] * (tokenLength - 1)
    addDict['nerToken'] = [ner] + ['nerX'] * (tokenLength - 1)
    addDict['tokenLength'] = tokenLength


    return addDict

#print(tokenizer.tokenize('I\'ll learn to swim in 12342 years.'))

# lists for sentences, tokens, labels, etc.
sentenceList = []
sentenceTokenList = []
posTokenList = []
nerTokenList = []
sentLengthList = []

# lists for BERT input
bertSentenceIDs = []
bertMasks = []
bertSequenceIDs = []

sentence = ''

# always start with [CLS] tokens
sentenceTokens = ['[CLS]']
posTokens = ['[posCLS]']
nerTokens = ['[nerCLS]']

with open('data/train_products.csv') as csv_file:
    csv_reader = reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        words = row[0].split(" ")
        tags  = row[2].split(" ")
        tags.pop()
        i = 0
        for word in words:
            ner = tags[i]
            if i == 0:
                sentenceLength = min(max_length -1, len(sentenceTokens))
                sentLengthList.append(sentenceLength)
                if sentenceLength >= max_length - 1:
                    sentenceTokens = sentenceTokens[:max_length - 2]
                    posTokens = posTokens[:max_length - 2]
                    nerTokens = nerTokens[:max_length - 2]
        
                sentenceTokens += ['[SEP]'] + ['[PAD]'] * (max_length -1 - len(sentenceTokens))
                posTokens += ['[posSEP]'] + ['[posPAD]'] * (max_length - 1 - len(posTokens) )
                nerTokens += ['[nerSEP]'] + ['[nerPAD]'] * (max_length - 1 - len(nerTokens) )
            
                sentenceList.append(sentence)

                sentenceTokenList.append(sentenceTokens)
                bertSentenceIDs.append(tokenizer.convert_tokens_to_ids(sentenceTokens))
                bertMasks.append([1] * (sentenceLength + 1) + [0] * (max_length -1 - sentenceLength ))
                bertSequenceIDs.append([0] * (max_length))
                             
                posTokenList.append(posTokens)
                nerTokenList.append(nerTokens)
        
                sentence = ''
                sentenceTokens = ['[CLS]']
                posTokens = ['[posCLS]']
                nerTokens = ['[nerCLS]']
                sentence += ' ' + words[0]
            else:
                addDict = addWord(word, ner)
                sentenceTokens += addDict['wordToken']
                #posTokens += addDict['posToken']
                nerTokens += addDict['nerToken']
            i = i + 1
        line_count = line_count + 1
sentLengthList = sentLengthList[2:]
sentenceTokenList = sentenceTokenList[2:]
bertSentenceIDs = bertSentenceIDs[2:]
bertMasks = bertMasks[2:]
bertSequenceIDs = bertSequenceIDs[2:]
#posTokenList = posTokenList[2:]
nerTokenList = nerTokenList[2:]


print(sentenceTokenList[2])
print(nerTokenList[2])
print(bertMasks[2])
print(bertSequenceIDs[2])

numSentences = len(bertSentenceIDs)

nerClasses = pd.DataFrame(np.array(nerTokenList).reshape(-1))
nerClasses.columns = ['tag']
nerClasses.tag = pd.Categorical(nerClasses.tag)
nerClasses['cat'] = nerClasses.tag.cat.codes
nerClasses['sym'] = nerClasses.tag.cat.codes
nerLabels = np.array(nerClasses.cat).reshape(numSentences, -1)

#divide in train and test

bert_inputs = np.array([bertSentenceIDs, bertMasks, bertSequenceIDs])

numSentences = len(bert_inputs[0])
np.random.seed(0)
training_examples = np.random.binomial(1, 0.7, numSentences)

trainSentence_ids = []
trainMasks = []
trainSequence_ids = []

testSentence_ids = []
testMasks = []
testSequence_ids = []

nerLabels_train =[]
nerLabels_test = []


for example in range(numSentences):
    if training_examples[example] == 1:
        trainSentence_ids.append(bert_inputs[0][example])
        trainMasks.append(bert_inputs[1][example])
        trainSequence_ids.append(bert_inputs[2][example])
        nerLabels_train.append(nerLabels[example])
    else:
        testSentence_ids.append(bert_inputs[0][example])
        testMasks.append(bert_inputs[1][example])
        testSequence_ids.append(bert_inputs[2][example])
        nerLabels_test.append(nerLabels[example])

X_train = np.array([trainSentence_ids,trainMasks,trainSequence_ids])
X_test = np.array([testSentence_ids,testMasks,testSequence_ids])

nerLabels_train = np.array(nerLabels_train)
nerLabels_test = np.array(nerLabels_test)

print(X_train[0,0])


nerDistribution = (nerClasses.groupby(['tag', 'cat']).agg({'sym':'count'}).reset_index()
                   .rename(columns={'sym':'occurences'}))

numNerClasses = nerDistribution.tag.nunique()


# Use a parameter pair k_start, k_end to look at slices. This helps with quick tests.

k_start = 0
#k_end = -1
k_end = 1000

if k_end == -1:
    k_end_train = X_train[0].shape[0]
    k_end_test = X_test[0].shape[0]
else:
    k_end_train = k_end_test = k_end



bert_inputs_train_k = [X_train[0][k_start:k_end_train], X_train[1][k_start:k_end_train],
                       X_train[2][k_start:k_end_train]]
bert_inputs_test_k = [X_test[0][k_start:k_end_test], X_test[1][k_start:k_end_test],
                      X_test[2][k_start:k_end_test]]


labels_train_k = nerLabels_train[k_start:k_end_train]
labels_test_k = nerLabels_test[k_start:k_end_test]


def custom_loss(y_true, y_pred):
    """
    calculate loss function explicitly, filtering out 'extra inserted labels'
    
    y_true: Shape: (batch x (max_length + 1) )
    y_pred: predictions. Shape: (batch x x (max_length + 1) x num_distinct_ner_tokens ) 
    
    returns:  cost
    """

    #get labels and predictions
    
    y_label = tf.reshape(tf.layers.Flatten()(tf.cast(y_true, tf.int32)),[-1])
    
    mask = (y_label < 3)   # This mask is used to remove all tokens that do not correspond to the original base text.

    y_label_masked = tf.boolean_mask(y_label, mask)  # mask the labels
    
    y_flat_pred = tf.reshape(tf.layers.Flatten()(tf.cast(y_pred, tf.float32)),[-1, numNerClasses])
    
    y_flat_pred_masked = tf.boolean_mask(y_flat_pred, mask) # mask the predictions
    
    return tf.reduce_mean(sparse_categorical_crossentropy(y_label_masked, y_flat_pred_masked,from_logits=False ))


O_occurences = nerDistribution.loc[nerDistribution.tag == 'O','occurences']
All_occurences = nerDistribution[nerDistribution.cat < 3]['occurences'].sum()

def custom_acc_orig_tokens(y_true, y_pred):
    """
    calculate loss dfunction filtering out also the newly inserted labels
    
    y_true: Shape: (batch x (max_length) )
    y_pred: predictions. Shape: (batch x x (max_length + 1) x num_distinct_ner_tokens ) 
    
    returns: accuracy
    """

    #get labels and predictions
    
    y_label = tf.reshape(tf.layers.Flatten()(tf.cast(y_true, tf.int64)),[-1])
    
    mask = (y_label < 3)
    y_label_masked = tf.boolean_mask(y_label, mask)
    
    y_predicted = tf.math.argmax(input = tf.reshape(tf.layers.Flatten()(tf.cast(y_pred, tf.float64)),\
                                                    [-1, numNerClasses]), axis=1)
    
    y_predicted_masked = tf.boolean_mask(y_predicted, mask)

    return tf.reduce_mean(tf.cast(tf.equal(y_predicted_masked,y_label_masked) , dtype=tf.float64))

def custom_acc_orig_non_other_tokens(y_true, y_pred):
    """
    calculate loss dfunction explicitly filtering out also the 'Other'- labels

    y_true: Shape: (batch x (max_length) )
    y_pred: predictions. Shape: (batch x x (max_length + 1) x num_distinct_ner_tokens )

    returns: accuracy
    """

    #get labels and predictions

    y_label = tf.reshape(tf.layers.Flatten()(tf.cast(y_true, tf.int64)),[-1])

    mask = (y_label < 2)
    y_label_masked = tf.boolean_mask(y_label, mask)

    y_predicted = tf.math.argmax(input = tf.reshape(tf.layers.Flatten()(tf.cast(y_pred, tf.float64)),\
                                                    [-1, numNerClasses]), axis=1)

    y_predicted_masked = tf.boolean_mask(y_predicted, mask)

    return tf.reduce_mean(tf.cast(tf.equal(y_predicted_masked,y_label_masked) , dtype=tf.float64))

#sess.close()

adam_customized = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.91, beta_2=0.999, epsilon=None, decay=0.1, amsgrad=False)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def ner_model(max_input_length, train_layers, optimizer):
    """
    Implementation of NER model

    variables:
        max_input_length: number of tokens (max_length + 1)
        train_layers: number of layers to be retrained
        optimizer: optimizer to be used

    returns: model
    """

    in_id = tf.keras.layers.Input(shape=(max_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_length,), name="segment_ids")


    bert_inputs = [in_id, in_mask, in_segment]

    bert_sequence = BertLayer(n_fine_tune_layers=train_layers)(bert_inputs)

    print(bert_sequence)

    dense = tf.keras.layers.Dense(256, activation='relu', name='dense')(bert_sequence)

    dense = tf.keras.layers.Dropout(rate=0.1)(dense)

    pred = tf.keras.layers.Dense(7, activation='softmax', name='ner')(dense)

    print('pred: ', pred)

    ## Prepare for multipe loss functions, although not used here

    losses = {
        "ner": custom_loss,
        }
    lossWeights = {"ner": 1.0
                  }

    model = tf.keras.models.Model(inputs=bert_inputs, outputs=pred)

    model.compile(loss=losses, optimizer=optimizer, metrics=[custom_acc_orig_tokens,
                                                          custom_acc_orig_non_other_tokens])


    model.summary()

    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


#Start session

sess = tf.Session()

model = ner_model(max_length + 1, train_layers=4, optimizer = adam_customized)

# Instantiate variables
initialize_vars(sess)

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.fit(
    bert_inputs_train_k,
    {"ner": labels_train_k },
    validation_data=(bert_inputs_test_k, {"ner": labels_test_k }),
    epochs=8,
    batch_size=32,
    #callbacks=[tensorboard]
)

# serialize model to JSON
model_json = model.to_json()
with open("models/bert_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/bert_model.h5")
print("Saved model to disk")



bert_inputs_infer = [X_test[0], X_test[1], X_test[2]]

#print(bert_inputs_infer)

result = model.predict(
    bert_inputs_infer,
    batch_size=32
)

print(result.shape)

print(np.argmax(result, axis=2)[6])

print(nerLabels_test[6])


