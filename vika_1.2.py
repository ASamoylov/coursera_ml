# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 05:00:41 2016

@author: Delirium
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt


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

data = pandas.read_csv('wines_quality.csv')
n = data.shape[0]

X = np.append(data.iloc[:, :-1].values.astype(np.float32), np.ones((n, 1)), axis=1)
y = data.iloc[:,-1].values.astype(np.float32)


m = X.shape[1]
np.random.seed(152)
rows = np.random.randint(len(X),size=m)
nm = X[rows,:]
rang = np.linalg.matrix_rank(nm)
print "Shape: " ,nm.shape
print "Rang: ", rang

check = np.array([True for j in range(0,len(nm))])
for i in range(0,len(nm)):
    check = np.array([True for j in range(0,len(nm))])
    check[i] = False
    tmp = nm[:-1,check]
    new_rang = np.linalg.matrix_rank(tmp)
    if new_rang == rang:
        print data.columns[i]
        last_f = i

check = np.array([True for j in range(0,len(nm))])
check[last_f] = False
X = X[:, check]

w0 = np.zeros(X.shape[1])
n_iter = 10000


q_h = dict()
for h in np.power(10.0, np.arange(-6, -2)):
    def const_step_it(iter):
        return h
    (w, qi) = grad_descent(X, y, w0, const_step_it, q_grad, n_iter)
    q_h.update({h: qi[-1]})

for (k,v) in sorted(q_h.items()):
    print "h=%f\tQ=%f" % (k,v)

(w, qi) = grad_descent(X,y,w0, const_step, q_grad, n_iter)



plt.plot(qi)
plt.ylabel('Q')
plt.xlabel('iter')

plt.show()




