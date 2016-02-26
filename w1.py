# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas

data = pandas.read_csv('titanic.csv', index_col='PassengerId')

print ("%i %i" % tuple(data['Sex'].value_counts()))

print ( "%i" %  data['Survived'])
