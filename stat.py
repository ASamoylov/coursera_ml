# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import scipy as sp

import pylab as plt

from scipy.stats import bernoulli


def get_plot(data, title, xlabel='value', ylabel='freq', bins=20):
    plt.hist(data, bins, normed=True, facecolor='blue', alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    
    plt.show()

def get_stat(data):
    avr = data.mean()
    med = np.median(data)
    disp = np.mean((data-data.mean())**2)
    sd = np.sqrt(disp)
    return avr, med, disp, sd
    

'''
for p in np.linspace(0.1,0.9,5):
    ber = bernoulli.rvs(p, size=500)
    avr = ber.mean()
    med = np.median(ber)
    disp = np.mean((ber-ber.mean())**2)
    sd = np.sqrt(disp)
    get_plot(ber, 'Bernoulli: p = {}'.format(p))
    print "p = %f\tmean: %f\tmedian: %f\tdisp: %f\tsqe: %f\n\n" % (p, avr, med, disp, sd)
''' 
    
'''
n=10
for p in np.linspace(0.1,0.9,5):
    binom = np.random.binomial(n,p, size=500)
    avr, med, disp, sd = get_stat(binom)
    get_plot(binom, 'Binomial: n = {} p = {}'.format(n,p))
    print "p = {}\tmean: {}\tmedian: {}\tdisp: {}\tsqe: {}\n\n".format(p, avr, med, disp, sd)
'''

for lam in np.linspace(1,10,4):
    pu = np.random.poisson(lam, size=500)
    avr, med, disp, sd = get_stat(pu)
    get_plot(pu, 'Poisson: $\lambda$ = {}'.format(lam), bins=50)
    print "lam = {}\tmean: {}\tmedian: {}\tdisp: {}\tsqe: {}\n\n".format(lam, avr, med, disp, sd)