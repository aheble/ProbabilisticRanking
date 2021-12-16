# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:35:24 2021

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

# Gibbs sampling (This may take a minute)
# number of iterations
num_iters = 100


# perform gibbs sampling, skill samples is an num_players x num_samples array
#mu, var = calc(m,std,50,3000,10)[0], [a**2 for a in calc(m,std,50,3000,10)[1]]

# EP ranking
# run message passing algorithm, returns mean and precision for each player
mps, pps, mcov, pcov = eprank(G, M, num_iters)
#print(mps[15],mps[0],mps[4],mps[10],pps[15],pps[0],pps[4],pcov[10])
#prob_skill = lambda i,j: 1 - stats.norm.cdf(0,np.mean(mcov[i])-np.mean(mcov[j]),math.sqrt(1+np.mean(pcov[i])+np.mean(pcov[j])))
prob_skill = lambda i,j: 1 - stats.norm.cdf(0,mps[i]-mps[j],math.sqrt((1/pps[i]) + (1/pps[j])))
post_skill = lambda i,j: 1 - stats.norm.cdf(0,mps[i]-mps[j],math.sqrt(1+(1/pps[i]) + (1/pps[j])))
#post_skill = lambda i,j:  stats.norm.cdf((np.mean(mcov[i])-np.mean(mcov[j]))/math.sqrt(1+np.mean(pcov[i])+np.mean(pcov[j])))
#post_skill = lambda i,j:  stats.norm.cdf((mps[i]-mps[j])/math.sqrt(1+pps[i]+pps[j]))
#print(prob_skill(mu[15], mu[0], p[15], p[0]))
#print(prob_skill(mu[15], mu[4], p[15], p[4]))
#print(prob_skill(mu[15], mu[10], p[15], p[10]))
#print(prob_skill(mu[0], mu[4], p[0], p[4]))
#print(prob_skill(mu[0], mu[10], p[0], p[10]))
#print(prob_skill(mu[4], mu[10], p[4], p[10]))
print(prob_skill(15,0))
print(prob_skill(15,4))
print(prob_skill(15,10))
print(prob_skill(0,4))
print(prob_skill(0,10))
print(prob_skill(4,10))

print(post_skill(15,0))
print(post_skill(15,4))
print(post_skill(15,10))
print(post_skill(0,4))
print(post_skill(0,10))
print(post_skill(4,10))
#print(post_skill(mu[15], mu[4], p[15], p[4]))
#print(post_skill(mu[15], mu[10], p[15], p[10]))
#print(post_skill(mu[0], mu[4], p[0], p[4]))
#print(post_skill(mu[0], mu[10], p[0], p[10]))
#print(post_skill(mu[4], mu[10], p[4], p[10]))