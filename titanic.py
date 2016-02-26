# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 13:04:32 2016

@author: asamoylov
"""

import numpy as np
import pandas 
import re
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('titanic.csv', index_col = 'PassengerId')

myf = open('tit1.txt', "w")
myf.write( "%i %i" % (data['Sex'].value_counts()[0], data['Sex'].value_counts()[1]))
myf.close()

myf = open('tit2.txt', "w")
eps = 100.*data['Survived'][data['Survived']==1].count()/data['Survived'].count()
myf.write("%.2f" % eps)
myf.close()

myf = open('tit3.txt', "w")
eps = 100. * data['Pclass'][data['Pclass']==1].count()/data['Pclass'].count()
myf.write("%.2f" % eps)
myf.close()

myf = open('tit4.txt', "w")
mn = data['Age'].mean()
med = data['Age'].median()
myf.write("%.2f %.2f" % (mn, med))
myf.close()

myf = open('tit5.txt', "w")
a = ((data['SibSp']-data['SibSp'].mean())*(data['Parch']-data['Parch'].mean())).sum()
b = ( ((data['SibSp']-data['SibSp'].mean())**2).sum() * ((data['Parch']-data['Parch'].mean())**2).sum())**0.5
r = a/b
myf.write("%.2f" % (r))
myf.close()


myf = open('tit6.txt', "w")
x = []
for s in data[data['Sex']=='female']['Name']:
    mm = re.search(r"(?:Mrs\..*\(|Miss\. )(?P<first_name>\w+)", s, re.I)
    if (mm):
        x.append(mm.group('first_name'))
df = pandas.Series(data=x).value_counts()
myf.write("%s" % df.idxmax())
myf.close()


data0 = data
X = np.array([[1, 2], [3, 4], [5, 6]])
y = np.array([0, 1, 0])
clf = DecisionTreeClassifier()
clf.fit(X, y)








