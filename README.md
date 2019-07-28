# Name entity recognition

## Introduction

For an ecommerce site, recognize brand names from titles using different algorithms. From basic ones such as comparing with list of brands to more complex ones such as bidirectional lstms with CRF.

## Instructions

1. Setup environment
```
pip install pipenv
pipenv --python 3.7
pipenv shell
pip install numpy
pip install keras
pip install tensorflow

```
2. Run product file parser
```
python parseProductsFile.py data/ecommerce_products.csv

```
3. Split the file into train and test (train will include validation)

```
python splitTrainTestProductFile.py

```
4. Run tokenization for words and chars with keras and pickle the output.
```
python train_char_tokenizer.py data/train_products.csv $max_num_chars_per_word $char_tokenizer_model_name $max_num_words_per_sentence
python train_word_tokenizer.py data/train_products.csv $max_num_words_per_sentence $word_tokenizer_model_name

```

5. Train networks
```
python train_network.py data/train_products.csv LSTM $max_num_words_per_sentence $max_num_chars_per_word $word_tokenizer_model_name $char_tokenizer_model_name
python train_network.py data/train_products.csv LSTMCRF $max_num_words_per_sentence $max_num_chars_per_word $word_tokenizer_model_name $char_tokenizer_model_name
python train_network.py data/train_products.csv CNNLSTMCRF $max_num_words_per_sentence $max_num_chars_per_word $word_tokenizer_model_name $char_tokenizer_model_name
python train_network.py data/train_products.csv BASELINE
```
