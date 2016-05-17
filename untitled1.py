# -*- coding: utf-8 -*-
"""
Created on Mon Apr 04 18:31:12 2016

@author: asamoylov
"""

import math


def fact(n):
    r=1
    for i in range(1,n+1):
        r = r * i
    return r

def soch(n,k):
    return fact(n)/(fact(k)*fact(n-k))



myf = open('test.txt', 'w')
myf2 = open('test2.txt', 'w')

s=list()

for i in range(10):
    s.append(str(i)+'\n')

myf.writelines(s)

myf.close()
myf2.close()