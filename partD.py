# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 17:02:45 2021

@author: abhin
"""
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from gibbsrank import gibbs_sample
from eprank import eprank
import pandas
import math
from cw2 import sorted_barplot
import scipy.stats as stats

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

# Gibbs sampling (This may take a minute)
# number of iterations
num_iters = 3000
# perform gibbs sampling, skill samples is an num_players x num_samples array
sps,mcov,pcov = gibbs_sample(G, M, num_iters) # random_nums

def probj(samples, burn, fin, thin, i, j):
    jarr = [[],[]]
    size = 1000000
    tot_i, tot_j = 0,0
    for a in range(burn,fin,thin):
        jarr[0].append(samples[i][a])
        jarr[1].append(samples[j][a])
    cov_mat = np.cov(jarr)
    mu_i = np.mean(samples[i][burn:fin:thin])
    mu_j = np.mean(samples[j][burn:fin:thin])
    new_samples = stats.multivariate_normal.rvs([mu_i, mu_j],cov_mat,size)
    for c, d in new_samples:
        if c>d: tot_i+=1
        else: tot_j+=1
    return tot_i/size, tot_j/size
    
pdn,pnd = probj(sps,50,3000,10,15,0)
pdf,pfd = probj(sps,50,3000,10,15,4)
pdm,pmd = probj(sps,50,3000,10,15,10)
pnf,pfn = probj(sps,50,3000,10,0,4)
pnm,pmn = probj(sps,50,3000,10,0,10)
pfm,pmf = probj(sps,50,3000,10,4,10)

print(pdn,pdf,pdm,pnd,pnf,pnm,pfd,pfn,pfm,pmd,pmn,pmf)