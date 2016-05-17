# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd

from __future__ import division

import re
from collections import Counter
from collections import defaultdict

def preprocess_message(text):
    text_lower = text.lower()
    words = re.sub("[^a-zA-Z]", " ", text_lower).split()
    return Counter(words)

def classes_prob(targets):
    class_probs = dict(Counter(targets))
    for key in class_probs.keys():
        class_probs[key] /= 1.0*len(targets)
    return class_probs

def find_counts(data):
    counts = defaultdict()
    for key in data['target'].unique():
        counts[key]=Counter()
    for _, (target, text) in data.iterrows():
        counts[target] = counts[target] + Counter(text)
    return counts


def find_probs(counts, alpha=0.1):
    tmp = []
    for k in counts.keys():
        tmp = tmp + counts[k].keys()
    uniq_words = np.unique(tmp)
    n_words = uniq_words.shape[0]
    probs = defaultdict()
    for k in counts.keys():
        probs[k]=defaultdict(lambda: 1./n_words)
    
    target_div = defaultdict()
    for k in counts.keys():
        target_div[k] = np.sum(counts[k][word] for word in counts[k]) + alpha*n_words
    
    for k in counts.keys():
        for word in uniq_words:
            probs[k][word] = (counts[k][word] + alpha) / target_div[k]
    
    return probs

def filter_bad_words(text):
    words = {}
    for word, count in text.iteritems():
        if word not in bad_words:
            words[word] = count
    return words

def classify(class_probs, probs, message):
    target_score=defaultdict()
    target_score_max = 0.
    target_max = ''
    for k in class_probs.keys():
        target_score[k]=class_probs[k]*np.prod([probs[k][word] for word in message])
        if target_score_max < target_score[k]:
            target_score_max = target_score[k]
            target_max = k
    return target_max

def classify_log(class_probs, probs, message):
    target_score=defaultdict()
    target_score_max = 0.
    target_max = class_probs.keys()[0]
    for k in class_probs.keys():
        target_score[k]=np.log(class_probs[k]) + np.sum([np.log(probs[k][word]) for word in message])
    
    target_max = target_score.keys()[0]
    target_score_max = target_score[target_max]
    for k in target_score.keys():
        if target_score_max < target_score[k]:
            target_score_max = target_score[k]
            target_max = k
    return target_max

def accuracy(y_true, y_pred):
    res = (np.array(y_true) == np.array(y_pred)).mean()
    return res

def count_freqs(messages):
    res = Counter()
    for mess in messages:
        res = res + Counter(mess)
    return res

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


class_probs = classes_prob(data_train.target)
print class_probs

small_data = pd.DataFrame([[1, {'cat': 1, 'eats': 1, 'a': 1, 'bird': 1}], 
                          [1, {'dog': 1, 'eats': 1, 'a': 1, 'bone': 1}],
                          [0, {'person': 1, 'eats': 1, 'a': 1, 'banana': 1}]], columns=['target', 'text'])

assert find_counts(small_data) == {0: defaultdict(int, {'a': 1, 'banana': 1, 'eats': 1, 'person': 1}),
                                  1: defaultdict(int, {'a': 2, 'bird': 1, 'bone': 1, 'cat': 1, 'dog': 1, 'eats': 2})}, "Failed!"

counts = find_counts(data_train)

print find_probs(find_counts(small_data))

probs = find_probs(counts, alpha=0.1)

assert classify({1: 2. / 3, 0: 1. / 3}, find_probs(find_counts(small_data)), {'cat': 1, 'eats': 1, 'a': 1, 'bone': 1}) == 1, "Failed"

make_classify = lambda message: classify(class_probs, probs, message)

assert accuracy([1, 2, 3], [3, 2, 1]) == 1. / 3, "Failed: [1, 2, 3], [3, 2, 1]"
assert accuracy([1, 2, 3], [1, 2, 3]) == 1., "Failed: [1, 2, 3], [1, 2, 3]"
assert accuracy([1, 2, 3], [2, 3, 1]) == 0, "Failed: [1, 2, 3], [2, 3, 1]"

print "Accuracy: ", accuracy(data_test.target, data_test.text.apply(make_classify))

print data_test[ data_test.text.apply(make_classify) == '' ].head()

assert classify_log({1: 2. / 3, 0: 1. / 3}, find_probs(find_counts(small_data)), {'cat': 1, 'eats': 1, 'a': 1, 'bone': 1}) == 1, "Failed"

make_classify_log = lambda m: classify_log(class_probs, probs, m)

print "Accuracy: ", accuracy(data_test.target, data_test.text.apply(make_classify_log))

#top 10 words
for k in probs.keys():
    print k
    for (key, val) in Counter(probs[k]).most_common()[:10]:
        print "\t{}\t{}".format(key,val)

print count_freqs(small_data.text)

words_counts = count_freqs(data_train.text)

bad_words = [i[0] for i in words_counts.most_common()[:10]]


def filter_bad_words(text):
    words = {}
    for word, count in text.iteritems():
        if word not in bad_words:
            words[word] = count
    return words

def filter_bad_words2(text):
    words = Counter()
    for word, count in text.iteritems():
        if word not in bad_words:
            word = word + Counter({word: count})
    return words

train_messages = data_train.apply(lambda row: filter_bad_words(row['text']), axis=1)
data_train.iloc[:, 1] = train_messages
test_messages = data_test.apply(lambda row: filter_bad_words(row['text']), axis=1)
data_test.iloc[:, 1] = test_messages

counts = find_counts(data_train)
good_probs = find_probs(counts, alpha=0.1)

make_classify_log = lambda m: classify_log(class_probs, good_probs, m)
print accuracy(data_test.target, data_test.text.apply(make_classify_log))
>>>>>>> origin/master
