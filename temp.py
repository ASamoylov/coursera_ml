# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import scipy as sp

import re
from collections import Counter

def preprocess_message(text):
    text_lower = text.lower()
    words = re.sub("[^a-zA-Z]", " ", text_lower).split()
    return Counter(words)

def classes_prob(targets):
    class_probs = dict(Counter(targets))
    for key in class_probs.keys():
        class_probs[key] /= 1.0*len(targets)
    return class_probs

data_train = pd.read_csv('data.train.csv', sep='\t')
data_test = pd.read_csv('data.test.csv', sep='\t')
#data_train.head()

print data_train.iloc[0,0], '\n', preprocess_message(data_train.iloc[0,1])


train_messages = data_train.apply(lambda row: preprocess_message(row['text']), axis=1)
data_train.iloc[:, 1] = train_messages
test_messages = data_test.apply(lambda row: preprocess_message(row['text']), axis=1)
data_test.iloc[:, 1] = test_messages



assert classes_prob([1, 1, 1]) == { 1: 1}, "Failed: 'classes_prob([1, 1, 1]) == { 1: 1}'"
assert classes_prob(['1', '1', '1']) == { '1': 1}, "Failed: 'classes_prob([\'1\', \'1\', \'1\']) == { \'1\': 1}'"
assert classes_prob([1, 1, 2, 2]) == { 1: 0.5, 2: 0.5}, "Failed: 'classes_prob([1, 1, 2, 2]) == { 1: 0.5, 2: 0.5}'"
assert classes_prob([1, 2, 3]) == { 1: 1. / 3, 2: 1. / 3, 3: 1. / 3}, "Failed: 'classes_prob([1, 2, 3]) == { 1: 1 / 3, 2: 1 / 3, 3: 1 / 3}'"
assert classes_prob([1, 2, 3, 1]) == { 1: 0.5, 2: 0.25, 3: 0.25}, "Failed: 'classes_prob([1, 2, 3, 1]) == { 1: 0.5, 2: 0.25, 3: 0.25}'"
assert classes_prob(['one', 'two']) == { 'one': 0.5, 'two': 0.5}, "Failed: 'classes_prob([\'one\', \'two\']) == { \'one\': 0.5, \'two\': 0.5}'"