# Name entity recognition - Brand Extraction

## Introduction

For an ecommerce site, recognize brand names from titles using different algorithms. From basic ones such as comparing with list of brands to more complex ones such as bidirectional lstms with CRF.

For tokenization and simple LSTM brand extraction models, ProductNER (https://github.com/etano/productner) was use as a baseline, but modified for the actual dataset, added CRF layer, and created CNN for char tokenization. 

For BERT model, W266 bert material was used as a baseline. It was modified for a different dataset and added a LSTM hidden layer to create a new model.

## Instructions

1. Setup environment
```
pip install pipenv
pipenv --python 3.7
pipenv shell
pip install numpy
pip install keras
pip install tensorflow
pip install git+https://www.github.com/keras-team/keras-contrib.git
pip install sklear

```
2. Get data
```
mkdir data
cd data
wget https://ner-files-w266.s3.amazonaws.com/ecommerce_products.zip
unzip ecommerce_products.zip
```
3. Get embeddings
```
cd data
wget http://dcc.uchile.cl/~jperez/word-embeddings/glove-sbwc.i25.vec.gz
gunzip glove-sbwc.i25.vec.gz

```
3. Run product file parser
```
python parseProductsFile.py data/ecommerce_products.csv
```
4. Split the file into train and test (train will include validation)

```
python splitTrainTestProductFile.py
```
5. Run tokenization for words and chars with keras and pickle the output.
```
python train_char_tokenizer.py data/train_products.csv $max_num_chars_per_word $char_tokenizer_model_name $max_num_words_per_sentence
python train_word_tokenizer.py data/train_products.csv $max_num_words_per_sentence $word_tokenizer_model_name

```
6. Train networks
```
#LSTM
python train_network.py data/train_products.csv LSTM $max_num_words_per_sentence $max_num_chars_per_word $word_tokenizer_model_name $char_tokenizer_model_name $embedding_dimension $embedding_file $epochs
#same but with params
python train_network.py data/train_products.csv LSTM 30 10 models/word_tokenizer models/char_tokenizer 300 glove-sbwc.i25.vec 8

#LSTM with CRF
python train_network.py data/train_products.csv LSTMCRF $max_num_words_per_sentence $max_num_chars_per_word $word_tokenizer_model_name $char_tokenizer_model_name $embedding_dimension $embedding_file $epochs
#same but with params
python train_network.py data/train_products.csv LSTMCRF 30 10 models/word_tokenizer models/char_tokenizer 300 glove-sbwc.i25.vec 8

#CNN FOR CHAR EMBEDDINGS + LSTM + CRF
python train_network.py data/train_products.csv CNNLSTMCRF $max_num_words_per_sentence $max_num_chars_per_word $word_tokenizer_model_name $char_tokenizer_model_name $embedding_dimension $embedding_file $epochs
#same but with params
python train_network.py data/train_products.csv CNNLSTMCRF 30 10 models/word_tokenizer models/char_tokenizer 300 glove-sbwc.i25.vec 8

python train_network.py data/train_products.csv BASELINE
```
7. Extract entitites
```
python extract_entities.py $model_dir $data_file LSTM $max_sequence_length_word $max_Sequence_lenght_char $prefix_word $prefix_char
python extract_entities.py models data/test_products.csv LSTM 30 10 models/word_tokenizer models/char_tokenizer

python extract_entities.py $model_dir $data_file LSTMCRF $max_sequence_length_word $max_Sequence_lenght_char $prefix_word $prefix_char
python extract_entities.py models data/test_products.csv LSTMCRF 30 10 models/word_tokenizer models/char_tokenizer

python extract_entities.py $model_dir $data_file CNNLSTMCRF $max_sequence_length_word $max_Sequence_lenght_char $prefix_word $prefix_char
python extract_entities.py models data/test_products.csv CNNLSTMCRF 30 10 models/word_tokenizer models/char_tokenizer

```

8. Train models.

To train the models jupyter notebook are added to the repository, where anyone can play with the adjustment parameters.

- Basemodel.ipynb
- bertNER.ipynb
- bertBLSTMCRFNER.ipynb
- CNNLSTMCRFNER.ipynb
- LSTMCRFNER.ipynb
- LSTMNER.ipynb
