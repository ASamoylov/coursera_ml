# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:45:29 2016

@author: asamoylov
"""

from math import sqrt, log

from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.snowball import Russian
from numpy import asarray, zeros
from numpy.linalg import svd

stemmer = LancasterStemmer()

class LSI(object):
    def __init__(self, stopwords, ignorechars, docs):
        self.wdict = {}
        self.dictionary = []
        self.stopwords = stopwords
        if type(ignorechars) == unicode: ignorechars = ignorechars.encode('utf-8')
        self.ignorechars = ignorechars

        self.docs = []
        for doc in docs: self.add_doc(doc)
        self.prepare()

    def prepare(self):
        self.build()
        self.calc()

    def dic(self, word, add=False):
        if type(word) == unicode: word = word.encode('utf-8')
        word = word.lower().translate(None, self.ignorechars)
        word = word.decode('utf-8')
        word = stemmer.stem(word)
        if word in self.dictionary:
            return self.dictionary.index(word)
        else:
            if add:
                self.dictionary.append(word)
                return len(self.dictionary) - 1
            else:
                return None

    def add_doc(self, doc):
        words = [self.dic(word, True) for word in doc.lower().split()]
        self.docs.append(words)
        for word in words:
            if word in self.stopwords:
                continue
            elif word in self.wdict:
                self.wdict[word].append(len(self.docs) - 1)
            else:
                self.wdict[word] = [len(self.docs) - 1]

    def build(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 0]
        self.keys.sort()
        self.A = zeros([len(self.keys), len(self.docs)])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i, d] += 1

    def calc(self):
        self.U, self.S, self.Vt = svd(self.A)

    def TFIDF(self):
        wordsPerDoc = sum(self.A, axis=0)
        docsPerWord = sum(asarray(self.A > 0, 'i'), axis=1)
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i, j] = (self.A[i, j] / wordsPerDoc[j]) * log(float(cols) / docsPerWord[i])

    def dump_src(self):
        print 'Здесь представлен расчет матрицы '
        for i, row in enumerate(self.A):
            print self.dictionary[i], row

    def print_svd(self):
        print 'Здесь сингулярные значения'
        print self.S
        print 'Здесь первые 3 колонки U матрица '
        for i, row in enumerate(self.U):
            print self.dictionary[self.keys[i]], row[0:3]
        print 'Здесь первые 3 строчки Vt матрица'
        print -1 * self.Vt[0:3, :]

    def find(self, word):
        idx = self.dic(word)
        if not idx:
            print 'слово невстерчается'
            return []
        if not idx in self.keys:
            print 'слово отброшено как не имеющее значения которое через stopwords'
            return []
        idx = self.keys.index(idx)
        print 'word --- ', word, '=', self.dictionary[self.keys[idx]], '.\n'
        # получаем координаты слова

        w = (-1 * self.U)[idx]
        return w
        # print 'word {}\t{}\t{}\n'.format(idx, w, word)
        # arts = []
        # x = -1 * self.Vt
        # for k, v in enumerate(self.docs):
        #     a = x[k]
        #     arts.append((k, v, a))
        # return arts