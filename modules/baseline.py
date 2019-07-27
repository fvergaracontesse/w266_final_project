import os
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle

class Baseline(object):

    def __init__(self, max_sequence_length=200, prefix="./models/brandDict"):
        self.prefix = prefix
        self.brands = []

    def save(self, prefix=None):

        pickle.dump(self.brands, open(self.prefix+".pickle", "wb"))

    def load(self, prefix=None):
        if prefix != None: self.prefix = prefix
        self.brands = pickle.load(open(self.prefix+".pickle", "rb"))

    def create(self, brands):
        for brand in brands:
            self.brands.append(brand)
        self.save()

