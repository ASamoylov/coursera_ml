# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 23:08:22 2016

@author: Delirium
"""

import numpy as np
import numpy
import pandas 
import sklearn.tree

data = pandas.read_csv('titanic.csv', index_col = 'PassengerId')

data1 = data[data['Age'].notnull()]

def sex_conv (x):
    if (x=='male'):
        return 1
    if (x=='female'):
        return 0
        
x = np.ndarray(shape=(0,4))
y = np.ndarray(shape=(0))

dat = data1.loc[:, ['Survived', 'Pclass', 'Fare', 'Age', 'Sex']]

for index, r in dat.iterrows():
    sur = r['Survived']
    cla = r['Pclass']
    far = r['Fare']
    age = r['Age']
    sex = sex_conv(r['Sex'])
    x = np.vstack((x,[cla,far,age,sex]))
    y = np.append(y,sur)
    
x = data.loc[data['Age'].notnull(),['Pclass', 'Fare', 'Age', 'Sex']]
y = data.loc[data['Age'].notnull(),['Survived']]

x['Sex'] = x['Sex'].apply(lambda sex: sex_conv(sex))
    
model = sklearn.tree.DecisionTreeClassifier(random_state=241)
model.fit(x,y)

importances = model.feature_importances_

print importances

print model.classes_

    