# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 20:53:00 2021

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
tot_d, size = 0, 0
for i in range(50,num_iters,10):
    if sps[15][i]>sps[0][i]: tot_d+=1
    size+=1

print("Djokovic: ", tot_d/size)