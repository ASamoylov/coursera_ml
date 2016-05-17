# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:21:58 2016

@author: Delirium
"""

def my_sum(a, b=0):
    c = 0
    if a > b:
        a = a+b
        b = a-b
        a = a-b
        
    for i in range(a, b + 1):
        c = c + i
        
    return c

def my_sin(x, eps=0.001):
    s=x
    ds=x
    i=1
    while (abs(ds)>eps):
        i=i+1
        ds = -1*ds * x**2/(2*(i-1) * (2*(i-1)+1))
        s = s + ds
    return s






print "Viberi deistvie:"
print "1. sin"
print "2. cos"
print "3. sum(a,b)"
choise = input("choise: ")

if (choise == 1):
    x = float(input("Input x: "))
    eps = input("Eps: ")
    print my_sin(x, eps)






