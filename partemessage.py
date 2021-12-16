# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 23:05:39 2021

@author: abhin
"""

import scipy.io as sio
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from gibbsrank import gibbs_sample
from eprank import eprank
import pandas
from cw2 import sorted_barplot
import math

def calc(m,v,burn,fin,thin):
    return (sum(m[burn:fin:thin])/len(m[burn:fin:thin]), sum(v[burn:fin:thin])/len(v[burn:fin:thin]))

# set seed for reproducibility
np.random.seed(0)
# load data,
data = sio.loadmat('tennis_data.mat')
# Array containing the names of each player
W = data['W']
# loop over array to format more nicely
for i, player in enumerate(W):
    W[i] = player[0]
# Array of size num_games x 2. The first entry in each row is the winner of game i, the second is the loser
G = data['G'] - 1
# Number of players
M = W.shape[0]
# Number of Games
N = G.shape[0]

# number of iterations
num_iters = 100


# EP ranking
# run message passing algorithm, returns mean and precision for each player
mps, pps, mcov, pcov = eprank(G, M, num_iters)
prob_skill = lambda i,j: 1 - stats.norm.cdf(0,mps[i]-mps[j],math.sqrt((1/pps[i]) + (1/pps[j])))
post_skill = lambda i,j: 1 - stats.norm.cdf(0,mps[i]-mps[j],math.sqrt(1+(1/pps[i]) + (1/pps[j])))

tests=np.zeros(M)

c=0
while c<10000:
    c+=1
    sams = [stats.norm.rvs(mps[i],math.sqrt(1+(1/pps[i]))) for i in range(M)]
    for b, index in enumerate(sorted(range(len(sams)), key=lambda k: sams[k])):
        tests[index]+=b/M
        
sorted_barplot(tests/c,W)