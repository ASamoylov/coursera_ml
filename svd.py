# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:58:10 2016

@author: asamoylov
"""

import numpy as np


a = np.random.randn(10000, 1000)

U, s, V = np.linalg.svd(a, full_matrices=True)

