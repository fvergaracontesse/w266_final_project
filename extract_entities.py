import sys, os, csv
import numpy as np
from operator import itemgetter
from modules.wordTokenizer import WordTokenizer
from modules.charTokenizer import CharTokenizer
from modules.LSTMNetwork import LSTMNetwork
from modules.LSTMCRFNetwork import LSTMCRFNetwork
from modules.CNNLSTMCRFNetwork import CNNLSTMCRFNetwork

def process(row, tokenizer, network):
    # Extract entities
    #print(row)
    data = tokenizer.tokenize([row['title']])
    tags = network.tag(data)[0]
    #print(tags)
    brand, brand_started = '', False
    for word, tag in zip(row['title'].split(' '), tags):
        max_tag = max(list(tag.items()), key=itemgetter(1))[0]
        if  'B-B' in max_tag and (not brand_started):
            brand = word
            brand_started = True
        elif 'I-B' in max_tag  and brand_started:
            brand += ' '+word
        else:
            brand_started = False
    row['brand'] = brand

    return row

def processCNN(row, tokenizer,charTokenizer, network):
    # Extract entities
    data = [tokenizer.tokenize([row['title']]),charTokenizer.tokenize([row['title']])]
    #print(row['title'])
    print(data[1])
    tags = network.tag(data)[0]
    #print(tags)
    brand, brand_started = '', False
    for word, tag in zip(row['title'].split(' '), tags):
        max_tag = max(list(tag.items()), key=itemgetter(1))[0]
        if  'B-B' in max_tag and (not brand_started):
            brand = word
            brand_started = True
        elif 'I-B' in max_tag  and brand_started:
            brand += ' '+word
        else:
            brand_started = False
    row['brand'] = brand

    return row

def main(argv):
    model_dir                                = sys.argv[1]
    data_file                                = sys.argv[2]
    max_sequence_length_word                 =int(sys.argv[4])
    max_sequence_length_char                 =int(sys.argv[5])
    prefix_word                              =sys.argv[6]
    prefix_char                              =sys.argv[7]


    wordTokenizer = WordTokenizer(max_sequence_length_word,prefix_word)
    wordTokenizer.load(os.path.join(model_dir, 'word_tokenizer'))

    charTokenizer = CharTokenizer(max_sequence_length_char,prefix_char,max_sequence_length_word)
    charTokenizer.load(os.path.join(model_dir, 'char_tokenizer'))

    #network params
    #data_dir                                   = './data/'
    #embedding_dim                              = int(sys.argv[7])
    #dropout_fraction                           = 0.5
    #hidden_dim                                 = 32
    #embedding_file                             = sys.argv[8]
    #epochs                                     = int(sys.argv[9])

    # Load named entity recognizer
    if sys.argv[3]=="LSTM":
        network = LSTMNetwork()
        network.load(os.path.join(model_dir, 'lstm'))
    elif sys.argv[3]=="LSTMCRF":
        network = LSTMCRFNetwork()
        network.load(os.path.join(model_dir, 'lstmCRF'))
    elif sys.argv[3]=="CNNLSTMCRF":
        network = CNNLSTMCRFNetwork()
        network.load(os.path.join(model_dir, 'CNNlstmCRF'))
        #charTokenizer = CharTokenizer()
        #charTokenizer.load(os.path.join(model_dir, 'char_tokenizer'))

    with open(data_file, 'r') as f:
        filename = 'processed_'+sys.argv[3]
        reader = csv.DictReader(f)
        outfile = open('.'.join(data_file.split('.')[:-1] + [filename, 'csv']), 'w')
        reader.fieldnames = ['title','brand','tags'] 
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames + ['ext_brand'])
        writer.writeheader()
        count = 0
        for row in reader:
            count += 1
            if sys.argv[3]=="CNNLSTMCRF":
                processed_row = processCNN(row, wordTokenizer,charTokenizer, network)
            else:
                processed_row = process(row, wordTokenizer, network)
            print(processed_row)
            writer.writerow(processed_row)

if __name__ == "__main__":
    main(sys.argv)

