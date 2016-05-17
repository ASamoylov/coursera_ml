# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 20:00:29 2016

@author: asamoylov
"""



n=14
m = 0
for i in range(n, 0, -1):
    m=m+i
    print m
    if (m+i-1 > 100):
        break