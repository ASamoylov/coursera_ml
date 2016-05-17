# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 14:37:33 2016

@author: Delirium
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from scipy.stats.stats import pearsonr

def Q(x, y, w):
    q = np.sum((np.sum(w*x,axis=1)-y)**2)
    return q

def q_grad(x, y, w):
    q = np.dot(2*(np.sum(w*x,axis=1)-y), x) / len(x)
    return q

def const_step(iter):
    return 0.00001

def decreasing_step(iter):
    return 0.000098/np.math.log(0.001*iter+3)

def grad_descent(x,y,w,step,grad,iters):
    q_i = [Q(x,y,w)]
    for i in range(iters):
        dw = - grad(x,y,w)*step(i)
        w = w + dw
        q_i.append(Q(x,y,w))
        if (np.linalg.norm(dw) < 1e-5):
            break
    return w, q_i



data = pd.read_csv('wines_quality.csv')
print data.head()

y = np.array(data['quality'].astype(np.float32))

x = data.iloc[:, :-1].values.astype(np.float32)

tmp = x[np.random.randint(0, x.shape[0],x.shape[1]), :]

rank_x = np.linalg.matrix_rank(tmp)

m=0
for i in range(x.shape[1]):
    true_m = np.array([True for j in range(x.shape[1])])
    true_m[i] = False
    #print true_m
    if (np.linalg.matrix_rank(tmp[:-1, true_m]) == rank_x):
        m=i

print 'fguskergbkeu rgse %i' % m
true_m = np.array([True for j in range(x.shape[1])])
true_m[m] = False

x=x[:,true_m]

iters = 10000
w0=np.zeros(x.shape[1])
w, q_i = grad_descent(x,y,w0, const_step, q_grad, iters)
plt.plot(q_i)
print Q(x,y,w)
'''
for i in range(x.shape[1]):
    mu = np.mean(x[:,i])
    dis = np.std(x[:,i])
    x[:,i] = (x[:,i]-mu)/dis
'''
w0=np.zeros(x.shape[1])
w, q_i = grad_descent(x,y,w0, const_step, q_grad, iters)
plt.plot(q_i)
print Q(x,y,w)

for i in range(x.shape[1]):
    z = x[:,i]
    pp = np.sum((z - np.mean(z))*(y - np.mean(y))) / np.sqrt(np.sum((z-np.mean(z))**2)*np.sum((y-np.mean(y))**2))
    print i, pp, pearsonr(x[:,i], y)


