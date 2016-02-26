# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 00:00:57 2016

@author: Delirium
"""

import pandas
import sklearn



data = pandas.read_csv('data-logistic.csv', names=['target', 'x1', 'x2'])

x = sklearn.preprocessing.scale(data[['x1', 'x2']])

sclr = sklearn.preprocessing.StandardScaler()

x = sclr.fit_transform(data[['x1', 'x2']])
