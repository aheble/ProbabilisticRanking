# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 18:52:45 2021

@author: abhin
"""
import scipy.stats as stats
import math

size = 1000000
tot_d, tot_n = 0,0
samples = stats.multivariate_normal.rvs([1.8997,1.4845],[[0.0469,0.0093],[0.0093,0.0375]],size)
for d, n in samples:
    if d>n: tot_d+=1
    else: tot_n+=1
print("Djokovic: ", tot_d/size)
print("Nadal: ", tot_n/size)
