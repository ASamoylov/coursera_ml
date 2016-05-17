# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 22:08:30 2016
@author: asamoylov
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

# data['Distance'].max() - это максимальная длина перелета
# data[ data['Distance'] == data['Distance'].max()] - это все рейсы с максимальной длинной перелета
# data[ data['Distance'] == data['Distance'].max()]['FlightNum'] - выберем из данных только номера рейсов
# Series.value_counts() - статистика по различным значениям в Series
print data[ data['Distance'] == data['Distance'].max() ]['FlightNum'].value_counts()
#14    31
#15    23
#Name: FlightNum, dtype: int64
# в итоге рейсы 14 и 15 максимально длинные 


# загрузим из файла информацию о аэропортах
aports = pd.read_csv('airports.csv')
print aports.columns
# определим iata самого частого аэропорта
aport = data['Origin'].describe()['top']
# aports[ aports['iata']==aport]['city'] - дает Series c городом
# выводим на экран Series.values - массив numpy.ndarray c данными
print 'Airport: %s\nCity: %s' % (data['Origin'].describe()['top'], aports[ aports['iata']==aport ]['city'].values[0])


# сгруппируем по аэропортам вылета, выберем время полета и посчитаем среднее. Результат - Series
a = data.groupby('Origin')['AirTime'].mean()
print "Max AirTime: %f" % a.max()
print "Airport with max airtime: %s" % a.idxmax()

# сгруппируем по аэропортам вылета, выберем аэропорт назначения и посчитаем статистику. Результат - Series
# индексом тут является пара (аэропорт, 'стандартное поле из describe')
# например ('ABE', 'count') или ('ACV', 'freq')
b = data.groupby('Origin')['Dest'].describe()
# выберем самое частотное для всех и число полетов
aport_origin = b[:, 'freq'].idxmax()
aport_origin_n = b[:, 'freq'].max()
# аэропорт назначения будет в поле 'top'
aport_dest = b[aport_origin, 'top']
print aport_origin,  aport_dest, aport_origin_n



c = data.groupby('Origin')
y = c.filter(lambda x: x['Distance'].count() > 1000)
z = y.groupby('Origin')['FlightNum', 'DepDelay']
for air, grp in z:
#    print air, grp['DepDelay']
    print air, 1.0*grp[ grp['DepDelay']> 0]['DepDelay'].count()/grp['DepDelay'].count()
    
    