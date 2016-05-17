# -*- coding: utf-8 -*-
"""
Created on Sun May 01 17:04:09 2016

@author: Delirium
"""

import numpy as np
import pandas as pd
import scipy as sp
import sklearn
from sklearn import feature_extraction
from sklearn.feature_extraction import DictVectorizer

enc = DictVectorizer()

data = pd.read_csv('salary-train.csv')

data['FullDescription'] = data['FullDescription'].str.lower().replace('[^a-zA-Z0-9]', ' ', regex = True)
data['LocationNormalized'].fillna('nan', inplace=True)
data['ContractTime'].fillna('nan', inplace=True)

#vectorizer = feature_extraction.text.TfidfVectorizer(min_df=5)
#x = vectorizer.fit_transform(data['FullDescription'])
y = data['SalaryNormalized']

X_train_categ = enc.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))