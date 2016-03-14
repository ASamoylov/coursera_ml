# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 14:06:53 2016

@author: asamoylov
"""

#%matplotlib inline

import time
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

data = pandas.read_csv('data-logistic.csv', header=None)
answers = data[0]
features = data[1:]



def update_weights(w1, w2, data, C):
    k = 0.1
    l = data.shape[0]
    w1 += k*1/l*sum([y*x1*(1 - 1/(1 + np.exp(-y*(w1*x1 + w2*x2)))) for index, (y, x1, x2) in data.iterrows()]) - k*C*w1
    w2 += k*1/l*sum([y*x2*(1 - 1/(1 + np.exp(-y*(w1*x1 + w2*x2)))) for index, (y, x1, x2) in data.iterrows()]) - k*C*w2
    return w1, w2

def gradient_descent(data, C=0):
    plt.ion()
    plt.show()
    w1, w2 = (0, 0)
    target_deviation = np.power(10.0, -5)
    deviation = 100
    iteration = 0
    l = data.shape[0]
    
    while deviation > target_deviation and iteration < 1000:
        new_w1, new_w2 = update_weights(w1, w2, data, C)
        plt.plot([w1, w2], [new_w1, new_w2])
        plt.draw()
        deviation = np.sqrt(np.power((new_w2 - w2), 2) + np.power((new_w1 - w1), 2))
        w1, w2 = new_w1, new_w2
        iteration += 1
    

    print 'Iteration number: {}'.format(iteration)
    return w1, w2


def get_prediction(w1, w2, data):
    return [1/(1 + np.exp(-w1*x1 - w2*x2)) for index, (y, x1, x2) in data.iterrows()]


ax = data[data[0] == 1].plot(kind='scatter', x=1, y=2, color='Blue', label='+1')
data[data[0] == -1].plot(kind='scatter', x=1, y=2, color='Red', label='-1', ax=ax);

print 'Regulation coefficient: 0'
w1, w2 = gradient_descent(data)
print 'Accuracy: {:.3}'.format(roc_auc_score(answers, get_prediction(w1, w2, data)))

print 'Regulation coefficient: 10'
w1, w2 = gradient_descent(data, C=10)
print 'Accuracy: {:.3}'.format(roc_auc_score(answers, get_prediction(w1, w2, data)))