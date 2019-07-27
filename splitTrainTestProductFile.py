import random
import math

# Configure paths to your dataset files here
DATASET_FILE = 'data/products.csv'
FILE_TRAIN = 'data/train_products.csv'
FILE_TESTS = 'data/test_products.csv'

# Make sure it adds to 100, no error checking below
PERCENT_TRAIN = 75
PERCENT_TESTS = 25

data = [l for l in open(DATASET_FILE, 'r')]

train_file = open(FILE_TRAIN, 'w')
tests_file = open(FILE_TESTS, 'w')

num_of_data = len(data)
num_train = int((PERCENT_TRAIN/100.0)*num_of_data)
num_tests = int((PERCENT_TESTS/100.0)*num_of_data)

data_fractions = [num_train, num_tests]
split_data = [[],[]]

rand_data_ind = 0

for split_ind, fraction in enumerate(data_fractions):
    for i in range(fraction):
        rand_data_ind = random.randint(0, len(data)-1)
        split_data[split_ind].append(data[rand_data_ind])
        data.pop(rand_data_ind)

for l in split_data[0]:
    train_file.write(l)
    
for l in split_data[1]:
    tests_file.write(l)
    
train_file.close()
tests_file.close()
