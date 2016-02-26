# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 18:10:21 2016

@author: asamoylov
"""

from sklearn.svm import SVC
import pandas

data = pandas.read_csv('svm-data.csv', names=['t', 'f1', 'f2'])

x_train = data[['f1','f2']].values
y_train = data['t'].values

model = SVC(C=100000, kernel='linear', random_state=241)

model.fit(x_train, y_train)

print model.support_+1

