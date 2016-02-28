# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 17:07:34 2016

@author: asamoylov
"""

import sklearn
from sklearn import datasets
import pandas
import numpy

data = sklearn.datasets.load_boston()


data.data = sklearn.preprocessing.scale(data.data)

kf = sklearn.cross_validation.KFold(n=len(data.data), n_folds=5, shuffle=True, random_state=42)

res = {'p':[], 'err': []}
for myp in numpy.linspace(1,10, num=200):
    model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski', p=myp)
    r = sklearn.cross_validation.cross_val_score(model, data.data, data.target, cv=kf, scoring='mean_squared_error').mean()
    res['p'].append(myp), res['err'].append(r)
    print p,'\t',r

