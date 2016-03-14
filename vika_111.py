# -*- coding: utf-8 -*-
"""
Created on Sat Mar 05 12:23:13 2016

@author: Delirium
"""

import numpy as np
import math



def sq_eq(a,b,c):
    d = b*b - 4*a*c
    x1 = (-b + math.sqrt(d))/(2*a)
    x2 = (-b - math.sqrt(d))/(2*a)
    return x1, x2
    


a,b = sq_eq(1, 3, -7)

z =  np.array([1,2,3,4,5], dtype=float)
r = np.array([2*np.random.ranf()+1 for i in range(0,15)], dtype = float)




def cycle(n, k):
    x = [i for i in range(1, k+1)]
    y = np.array([x[i%k] for i in range(0,n)])
    return y


print cycle(10,3)

def next_diagonal (x):
    n = len(x)
    m = n+1
    a = [0]* (n+1)*(n+1)
    for i in range(len(x)) :
        a[i*m+i+1] = x[i]
    
    print a
    print np.reshape(a,(n+1, n+1))

next_diagonal(np.arange(1,5))

def get_elements(x, i, j):
    n = x.shape[1]
    m = x.shape[0]
    k = i.shape[0]
    print n,m,k
    for i0 in range(0, k):
        print "x[%i, %i]: %i" % (i[i0], j[i0],x[i[i0], j[i0]])
        pass
    
    y = np.array([x[i[i0], j[i0]] for i0 in range(0, k)])
    z = x[i,j]
    return y

n=10
x = np.random.randint(1000, size=(n,n))
i = np.random.randint(n, size=15)
j = np.random.randint(n, size=15)
print x
    
dfgdfg = get_elements(x, i, j)

print dfgdfg

def min_odd(x):
    return min(x[x%2 != 0])


print min_odd(x)
print ''
print ''

def nearest_value(x, v):
    c = np.abs(v - x)
    y = x[ c == min(c.flatten()) ]
    return y[0]

v = 5
print nearest_value(x, v)





