# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 21:04:22 2016

@author: Delirium
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_linear_data(n):
    np.random.seed(42)
    x = np.linspace(0, 10, n) + np.random.normal(0, 3, n)
    y = 2 * x + 5 + np.random.normal(0, 2, n)
    return x, y


def Q(x, y, w):
    q = np.sum((np.sum(w*x,axis=1)-y)**2)
    return q

def q_grad(x, y, w):
    q = np.dot(2*(np.sum(w*x,axis=1)-y), x) / len(x)
    return q

def const_step(iter):
    return 0.01

def decreasing_step(iter):
    return 0.01/(iter+1)

def grad_descent(x,y,w,step,grad,iters):
    q_i = [Q(x,y,w)]
    for i in range(iters):
        dw = - grad(x,y,w)*step(i)
        w = w + dw
        q_i.append(Q(x,y,w))
        if (np.linalg.norm(dw) < 1e-5):
            break
    return w, q_i
    
    
  
(x,y) = generate_linear_data(200)

w0 = np.array([0,0])
x_train = np.array([x, np.ones(len(x))]).T
y_train = y

print Q(x_train, y_train, w0)

w, q_i = grad_descent(x_train, y_train, w0, const_step, q_grad, 10000)

#w, q_i = grad_descent(x_train, y_train, w0, decreasing_step, q_grad, 10000)

plt.plot(q_i)
plt.ylabel('Q')
plt.xlabel('iter')
plt.ylim(0, 3000)
plt.show()

x_g = np.linspace(-5, 15, 20)
x_tmp = np.array([x_g, np.ones(len(x_g))]).T
y_g = np.sum(w*x_tmp, axis=1)

plt.plot(x,y,'go', x_g, y_g, 'b-')
plt.show()




