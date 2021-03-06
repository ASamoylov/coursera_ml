# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 00:00:57 2016

@author: Delirium
"""

import pandas
import sklearn
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.metrics import accuracy_score




def Q(x,y,w):
    q = np.sum((np.sum(w*x, axis=1) - y)**2)
    return q
    
def q_grad(x,y,w):
    dq = np.dot( (np.sum(w*x, axis=1) - y), x ) / len(x)
    return dq

def const_step(iter):
    return 0.0001
    
def decreasing_step(iter):
    return const_step(iter)/np.log2(2+iter)

def grad_descent(x,y,w, step, grad, iters):
    qs = [Q(x,y,w)]
    for i in range(iters):
        dw = - step(i)*grad(x,y,w)
        w_new = w + dw
        w = w_new
        qs.append(Q(x,y,w))
        if np.linalg.norm(dw) < 1e-5:
            break
    return w, qs

# *****************************
def Q_lg(x,y,w):
    q = np.sum( np.log( 1+np.exp(-y*np.sum(w*x, axis=1)) )) / len(x)
    return q

def q_grad_lg(x,y,w):
    tmp = np.dot( - y*(  1- 1 / ( 1 + np.exp( -y*np.sum(w*x, axis=1)) )  ), x)
    dq = tmp / len(x)
#    x1=x[:,0]
#    x2=x[:,1]

#    w1 = sum((1-1/(1+np.exp(-y*np.sum(w*x, axis=1))))*y*x1) / len(x)
#    w2 = sum((1-1/(1+np.exp(-y*np.sum(w*x, axis=1))))*y*x2) / len(x)
    return dq

def q_grad2_lg(x,y,w):
    dq=np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        tmp=0
        for j in range(x.shape[0]):
            tmp = tmp + (np.sum(w*x[j,:]) - y[j])*x[j,i]
        dq[i] = tmp / len(x)
        
    #dq = np.dot( (np.sum(w*x, axis=1) - y), x ) / len(x)
    return dq

def grad_descent_l2(x,y,w, k, c, grad, iters):
    qs = [Q_lg(x,y,w)]
    for i in range(iters):
        dw = -k*grad(x,y,w) - k*c*w
        w = w + dw 
        qs.append(Q_lg(x,y,w))
        if np.linalg.norm(dw) < 1e-5:
            break
    return w, qs

column_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'target']
data = pandas.read_csv('train.csv', names=column_names)
sclr = sklearn.preprocessing.StandardScaler()

#x = sclr.fit_transform(data[['x1', 'x2']])
x0 = data[[str(i+1) for i in range(10)]].values

x = sclr.fit_transform(data[[str(i+1) for i in range(10)]])

#kfold = KFold(x.shape[0], n_folds=10, shuffle=True, random_state=42)
#scores = cross_val_score(LogisticRegression(random_state=42, C=10.0, penalty='l2'), x, y, cv=kfold, scoring='accuracy')
#print np.mean(scores), np.std(scores)

model = LogisticRegression(random_state=42, C=10.0, penalty='l2')
model.fit(x,y)
#model.score

y = data['target'].values
w0 = np.zeros(x.shape[1])

(w,qi) = grad_descent_l2(x,y,w0, 0.1, 0, q_grad_lg , 10000)

a = 1 / (1+np.exp(-np.sum(w*x, axis=1)))
print "%.5f" % sklearn.metrics.roc_auc_score(y,a)

plt.plot(qi)
plt.show()



