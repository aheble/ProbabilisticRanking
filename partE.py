# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 21:51:49 2021

@author: abhin
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from gibbsrank import gibbs_sample
from eprank import eprank
import pandas
from cw2 import sorted_barplot
import math
import scipy.stats as stats

# set seed for reproducibility
np.random.seed(0)
# load data,
data = sio.loadmat('tennis_data.mat')
# Array containing the names of each player
W = data['W']
# loop over array to format more nicely
players = []
for i, player in enumerate(W):
    W[i] = player[0]
    players.append(player[0])
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
skill_samples, mcov1, pcov1 = gibbs_sample(G, M, num_iters) # random_nums
means=[]
samples=np.zeros((M,295))

tests=np.zeros(M)
for i,s in enumerate(skill_samples):
    means.append(np.mean(s[50::10]))
    samples[i]=s[50::10]
cov = np.cov(samples)+np.identity(M)
c=0
while c<10000:
    c+=1
    sams = stats.multivariate_normal.rvs(means,cov)
    for b, index in enumerate(sorted(range(len(sams)), key=lambda k: sams[k])):
        tests[index]+=b/M
        
sorted_barplot(tests/c,W)
    
# EP ranking
#num_iters = 200
# run message passing algorithm, returns mean and precision for each player
#mean_player_skills, precision_player_skills, mcov2, pcov2 = eprank(G, M, num_iters)
#print(pcov)