# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 17:46:40 2016

@author: asamoylov
"""

from math import sqrt

from LSI import LSI
import glob

def distance(c1, c2):
    if len(c1) != len(c2):
        raise Exception('Length mismatch')
    d = [(c1[i]-c2[i])**2 for i in xrange(len(c1))]
    return sqrt(sum(d))

def getContent(path):
    with open(path, "r") as f:
        return f.read()

root = '/home/luckybug/Документы/Reuters21578-Apte-90Cat/training/'

filenames = [f for f in glob.iglob(root + '/**/*')][:700]
docs = [getContent(f) for f in filenames]

print("docs ready")

lsi = LSI(stopwords=[" ", ",", "."], ignorechars="", docs=docs)

print("lsi created")

#lsi.print_svd()

c1 = lsi.find("Exchange")
c2 = lsi.find("Mercantile")
c3 = lsi.find("Indonesia")



e = 100
print(distance(c1[:e], c2[:e]))
print(distance(c2[:e], c3[:e]))
print(distance(c3[:e], c1[:e]))

e = 10
print(distance(c1[:e], c2[:e]))
print(distance(c2[:e], c3[:e]))
print(distance(c3[:e], c1[:e]))