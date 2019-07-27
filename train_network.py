"""Script to train a product category ner based on product titles and descriptions
"""

import sys, csv
from modules.tokenizer import WordTokenizer
from modules.charTokenizer import CharTokenizer
from modules.LSTMNetwork import LSTMNetwork
from modules.LSTMCRFNetwork import LSTMCRFNetwork
from modules.CNNLSTMCRFNetwork import CNNLSTMCRFNetwork
from modules.baseline import Baseline

MAX_TEXTS = 1000000

def main(argv):
    # Fetch data
    modelo = sys.argv[2]

    texts, tags, brands = [], [], []
    with open(sys.argv[1], 'r') as f:
        reader = csv.DictReader(f, fieldnames=["title","brand","tags"])
        count = 0
        for row in reader:
            #print(row)
            count += 1
            text, tag_set = row['title'], row['tags'].split(' ')[:-1]
            texts.append(text)
            tags.append(tag_set)
            brands.append(row['brand'])
            if count >= MAX_TEXTS:
                break
    print(('Processed %s texts.' % len(texts)))

    # Tokenize texts
    tokenizer = WordTokenizer()
    tokenizer.load()
    data = tokenizer.tokenize(texts)
    charTokenizer = CharTokenizer()
    charTokenizer.load()
    charData = charTokenizer.tokenize(texts)

    # Get labels from NER
    if modelo == "LSTM":
        network = LSTMNetwork()
        labels = network.get_labels(tags)
        # Compile NER network and train
        network.compile(tokenizer)
        network.train(data, labels, epochs=2)
    elif modelo == "LSTMCRF":
        network = LSTMCRFNetwork()
        labels = network.get_labels(tags)
        # Compile NER network and train
        network.compile(tokenizer)
        network.train(data, labels, epochs=2)
    elif modelo == "CNNLSTMCRF":
        network = CNNLSTMCRFNetwork()
        labels = network.get_labels(tags)
        # Compile NER network and train
        network.compile(tokenizer,charTokenizer)
        network.train([data,charData], labels, epochs=2)
    elif modelo == "BASELINE":
        baseline = Baseline()
        baseline.create(brands)
        


if __name__ == "__main__":
    main(sys.argv)

