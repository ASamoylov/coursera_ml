# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 00:00:57 2016

@author: Delirium
"""

import pandas
import sklearn
import numpy as np
import matplotlib.pyplot as plt


def Q(x,y,w):
    q = np.sum((np.sum(w*x, axis=1) - y)**2)
    return q

def q_grad(x,y,w):
    dq = np.dot( (np.sum(w*x, axis=1) - y), x ) / len(x)
    return dq

def q_grad_lg(x,y,w):
    tmp = np.dot(y*(  1-1/( 1+np.exp(-y*np.sum(w*x, axis=1)) )  ), x)
    dq = - tmp / len(x)
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

def grad_descent_l2(x,y,w, k, c, grad, iters):
    qs = [Q(x,y,w)]
    for i in range(iters):
        dw = -k*grad(x,y,w) - k*c*w
        w_new = w + dw 
        w = w_new
        qs.append(Q(x,y,w))
        if np.linalg.norm(dw) < 1e-5:
            break
    return w, qs


data = pandas.read_csv('data-logistic.csv', names=['target', 'x1', 'x2'])

sclr = sklearn.preprocessing.StandardScaler()

x = sclr.fit_transform(data[['x1', 'x2']])
y = data['target'].values
w0 = np.zeros(x.shape[1])

(w,qi) = grad_descent_l2(x,y,w0, 0.001, 0.001, q_grad , 10000)
a = 1 / (1+np.exp(-np.sum(w*x, axis=1)))
print sklearn.metrics.roc_auc_score(y,a)


plt.plot(qi)