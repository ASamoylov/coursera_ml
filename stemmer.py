# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 18:10:41 2016

@author: asamoylov
"""

from nltk.stem.snowball import RussianStemmer

mystem = RussianStemmer()

str0 = "поезд"

print mystem.stem(str0.decode("utf-8"))


