# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:15:31 2016

@author: Delirium
"""

import pandas as pd
import numpy as np


data = pd.read_csv('2008.csv')

#самая частая причина отмены рейса
# data['CancellationCode'] - дает объект Series
# Series.describe() - дает некоторую статистику, для нечисловых данных top - поле, которые хранит самый частое значение
print data['CancellationCode'].describe()
print data['CancellationCode'].describe()['top']
# 'B' - погода

#Максимальное время задержки из-за погоды
print data['WeatherDelay'].max()
#797.0

# среднее минимальное и максимальное расстояние для самолета
print "Среднее: %f" % data['Distance'].mean()
print "Минимум: %f" % data['Distance'].min()
print "Максимум: %f" % data['Distance'].max()





