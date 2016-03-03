# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 00:43:50 2016

@author: Delirium
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_linear_data(n):
    np.random.seed(42)
    x = np.linspace(0, 10, n) + np.random.normal(0, 3, n)
    y = 2 * x + 5 + np.random.normal(0, 2, n)
    return x, y
     
def Q(x,y,w):
    q = np.sum((np.sum(w*x, axis=1) - y)**2)
    return q

def q_grad(x,y,w):
    dq = np.dot( (np.sum(w*x, axis=1) - y), x ) / len(x)
    return dq

def const_step(iter):
    return 0.01
    
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

(x,y) = generate_linear_data(200)

w0 = np.array([0,0])
x_train = np.array([x, np.ones(len(x))]).T
y_train = y

n_iter = 10000

q_h = dict()
for h in np.power(10.0, np.arange(-3, 1)):
    def const_step_it(iter):
        return h
    (w, qi) = grad_descent(x_train, y_train, w0, const_step_it, q_grad, n_iter)
    q_h.update({h: qi[-1]})

for (k,v) in sorted(q_h.items()):
    print "h=%f\tQ=%f" % (k,v)

print "decreasing_step_it:"
def decreasing_step_it(iter):
    return 1/np.log2(iter+100.0)

(w, qi) = grad_descent(x_train, y_train, w0, decreasing_step_it, q_grad, n_iter)
print qi[-1]
print w

    
(w, qi) = grad_descent(x_train, y_train, w0, const_step, q_grad, n_iter) 

print qi[-1]
print w

plt.plot(qi)
plt.ylabel('Q')
plt.xlabel('iter')
plt.ylim(0, 3000)
plt.show()

x_g = np.linspace(-5, 15, 20)
x_tmp = np.array([x_g, np.ones(len(x_g))]).T
y_g = np.sum(w*x_tmp, axis=1)

plt.plot(x,y,'go', x_g, y_g, 'b-')
plt.show()


#########################

import pandas

data = pandas.read_csv('wines_quality.csv')



