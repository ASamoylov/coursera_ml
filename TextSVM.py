# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:38:03 2016

@author: Delirium
"""
import sklearn
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn import grid_search
import numpy as np
import pandas


newgroups = datasets.fetch_20newsgroups(subset = 'all', categories=['alt.atheism', 'sci.space'])

vectorizer = sklearn.feature_extraction.text.TfidfVectorizer()
x = vectorizer.fit_transform(newgroups.data)
y = newgroups.target


model = SVC(C=1000, kernel='linear', random_state=241)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(x, y)

best_c = gs.best_params_['C']

b_model = SVC(C=best_c, kernel='linear', random_state=241)
b_model.fit(x,y)

my_list = [vectorizer.get_feature_names()[i] for i in pandas.Series(b_model.coef_.toarray().reshape(-1)).abs().nlargest(10).index.values]
print " ".join(sorted(my_list))

myf = open('TextSVM.txt', 'w')
myf.write(" ".join(sorted(my_list)))
myf.close()

