# -*- coding: utf-8 -*-
"""
Created on Mon Feb 08 23:10:04 2016

@author: Delirium
"""

import pandas
#import sklearn
import sklearn
from sklearn.neighbors import KNeighborsClassifier
import numpy
sd = ['typ', '1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
data = pandas.read_csv('wine.data.txt', names=sd)

wine_target = data['typ']
wine_data = data[sd[1:]]

kf = sklearn.cross_validation.KFold(n=len(wine_data), n_folds=5, shuffle=True, random_state=42)

'''
for trn, tst in kf:
    x_trn, x_tst = wine_data.loc[trn, :], wine_data.loc[tst, :]
    y_trn, y_tst = wine_target[trn], wine_target[tst]
    
    k=1
    model = sklearn.neighbors.KNeighborsClassifier(k)
    model.fit(x_trn,y_trn)
    print model.score(x_tst, y_tst)
'''
    
res = {'in':[], 'r':[]}
for k in range(1,51):
    model = sklearn.neighbors.KNeighborsClassifier(k)
    r = sklearn.cross_validation.cross_val_score(model, wine_data, wine_target, cv=kf).mean()
    res['in'].append(k), res['r'].append(r)
#    print "%i\t%f" % (k,r)
    
d = pandas.DataFrame(res)
print d[ d['r']==d['r'].max()][['in', 'r']]

wine_data = sklearn.preprocessing.scale(wine_data)
res = {'in':[], 'r':[]}
for k in range(1,51):
    model = sklearn.neighbors.KNeighborsClassifier(k)
    r = sklearn.cross_validation.cross_val_score(model, wine_data, wine_target, cv=kf).mean()
    res['in'].append(k), res['r'].append(r)
#    print "%i\t%f" % (k,r)
    
d = pandas.DataFrame(res)

print d[ d['r']==d['r'].max()][['in', 'r']]
