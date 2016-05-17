# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:03:07 2016

@author: asamoylov
"""

import sys

def decor(fun):
    def ret_dec(a,b):
        a=a*2
        return fun(a,b)
    return ret_dec

def my_fun_sum(x1,x2):
    return x1+x2

@decor
def my_fun_sum2(a,b):
    return a*b


print my_fun_sum(1,2)
print my_fun_sum2(2,3)