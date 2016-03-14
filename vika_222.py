# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:15:31 2016

@author: Delirium
"""

import pandas as pd
import numpy as np


#data = pd.read_csv('my2008.csv')

print data['CancellationCode'].value_counts()

n_d = dict()

for index, n in data['Origin'].iterrows():
    data[ data['Origin']]
        n_d.update({n: d})
        
m = 0
num = 0
for k in n_d.keys():
    if m < n_d[k]:
        m = n_d[k]
        num = k
    
    



