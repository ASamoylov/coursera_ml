# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 21:31:47 2016

@author: asamoylov
"""
import matplotlib.pyplot as plt
import numpy as np






def generate_linear_data(n):
    np.random.seed(42)
    x = np.linspace(0, 10, n) + np.random.normal(0,3, n)
    y = 2 * x + 5 + np.random.normal(0,2,n)
    return x, y
    
def q_grad(x,y,w):
    q = np.sum((np.sum(w*x, axis=1) - y)**2) / x.shape[0]
    return q
    
    

(a,b) = generate_linear_data(100)
one = np.ones(a.shape[0])

x_tr = np.array([a,one]).T
y_tr = b


plt.scatter(a,b)




