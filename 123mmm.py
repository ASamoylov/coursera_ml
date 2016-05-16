# -*- coding: utf-8 -*-
"""
Created on Sat Apr 02 13:34:57 2016

@author: Delirium
"""
def fact(n):
    r = 1
    for i in range(1, n+1):
        r=r*i
    return r

def soch(n, k):
    return 1.*fact(n)/(fact(k) * fact(n-k))

sum0 = 0
pr=1
for i in range(0, 9, 2):
    sum0 = sum0+soch(9,i)
    print i, soch(9,i), sum0

print 'srthsrth'

for i in range(1, 10, 2):
    sum0 = sum0+soch(9,i)
    print i, soch(9,i), sum0
    
print 'kjervguoaerg'

for i in range(1, 10):
    n= soch(9,i)*(2**(9-i))
    sum0=sum0+n
    print i, n, sum0
    
    
def nod(a,b):
    while a!=0 and b!=0:
        if a > b:
            a = a % b
        else:
            b = b % a
    return a+b
    
def razn(n):
    s = list(str(n))
    for i in s:
        if s.count(i) > 1:
            return 0
    return 1
 
print "********************"
for h in range(1,10):
    for u in range(0,10):
        for n in range(1,10):
            for a in range(0,10):
                x=h*100 + u*10 +h
                y = n*100 +a*10 + n
                if (nod(x,y) == 1 and x/y < 1):
                    z1 = int (10000.0*x/y)
                    z2 = int(100000000.0*x/y) % 10000
                    if (z1==z2 and razn(z1)):
                        print x,"  ",y, "    ", z1, "  " ,z2