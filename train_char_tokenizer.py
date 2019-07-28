"""Script to train a word tokenizer
"""

import sys, csv
from modules.charTokenizer import CharTokenizer

MAX_TEXTS = 1000000

def main(argv):
    if len(argv) >= 2:

        # Fetch data
        texts = []
        with open(sys.argv[1], 'r') as f:
            reader = csv.DictReader(f, fieldnames=["title","brand"])
            count = 0
            for row in reader:
                count += 1
                text = row['title']
                texts.append(text)
                if count >= MAX_TEXTS:
                    break
        print(('Processed %s texts.' % len(texts)))

        # Tokenize texts
        #set params:
        max_sequence_length                 =sys.argv[2]
        prefix                              =sys.argv[3]
        max_sequence_sentence_words         =sys.argv[4]

        charTokenizer = CharTokenizer(max_sequence_length,prefix,max_sequence_sentence_words)
        charTokenizer.train(texts)

if __name__ == "__main__":
    main(sys.argv)
