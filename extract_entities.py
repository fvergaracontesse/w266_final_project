import sys, os, csv
import numpy as np
from operator import itemgetter
from modules.tokenizer import WordTokenizer
from modules.charTokenizer import CharTokenizer
from modules.LSTMNetwork import LSTMNetwork
from modules.LSTMCRFNetwork import LSTMCRFNetwork
from modules.CNNLSTMCRFNetwork import CNNLSTMCRFNetwork

def process(row, tokenizer, network):
    # Extract entities
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
    model_dir = sys.argv[1]
    data_file = sys.argv[2]

    # Load tokenizer
    tokenizer = WordTokenizer()
    tokenizer.load(os.path.join(model_dir, 'tokenizer'))

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
        charTokenizer = CharTokenizer()
        charTokenizer.load(os.path.join(model_dir, 'charTokenizer'))

    with open(data_file, 'r') as f:
        filename = 'processed_'+sys.argv[3]
        reader = csv.DictReader(f)
        outfile = open('.'.join(data_file.split('.')[:-1] + [filename, 'csv']), 'w')
        print(reader.fieldnames)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames + ['ext_brand'])
        writer.writeheader()
        count = 0
        for row in reader:
            count += 1
            if sys.argv[3]=="CNNLSTMCRF":
                processed_row = processCNN(row, tokenizer,charTokenizer, network)
            else:
                processed_row = process(row, tokenizer, network)
            print(processed_row)
            writer.writerow(processed_row)

if __name__ == "__main__":
    main(sys.argv)

