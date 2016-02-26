# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 20:03:01 2016

@author: asamoylov
"""

import sklearn
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
#import numpy
import pandas

data = pandas.read_csv('perceptron-train.csv', names = ['target', 'f1', 'f2'])
cntrl = pandas.read_csv('perceptron-test.csv', names = ['target', 'f1', 'f2'])

myp = Perceptron(random_state=241)

x_train = data[['f1', 'f2']].values
y_train = data['target'].values

x_test = cntrl[['f1', 'f2']].values
y_test = cntrl['target']

myp.fit(x_train, y_train)

y_out = myp.predict(x_test)

accur1 = sklearn.metrics.accuracy_score(y_test, y_out)

sclr = StandardScaler()

x_train_n = sclr.fit_transform(x_train)
x_test_n =  sclr.transform(x_test)

myp_n = Perceptron(random_state=241)
myp_n.fit(x_train_n, y_train)
y_out_n = myp_n.predict(x_test_n)

accur2 = sklearn.metrics.accuracy_score(y_test, y_out_n)

print "not norm: %.4f \tnorm: %.4f \tdiff: %.4f" % (accur1, accur2, accur2-accur1)

myf = open('perceptron_out.txt', 'w')
myf.write("%.3f" % (accur2-accur1))
myf.close()







