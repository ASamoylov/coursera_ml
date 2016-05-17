# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 21:30:08 2016

@author: Delirium
"""

def myf(n,m,d):
    if n>=m:
        print n
        return 1./n
    else:
        print n
        return 1./(n+myf(n+d,m,d))

print myf(1,103,2)