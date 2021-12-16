# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 13:14:19 2021

@author: abhin
"""

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from gibbsrank import gibbs_sample
from eprank import eprank
import pandas
from cw2 import sorted_barplot

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

# EP ranking
num_iters = 100
# run message passing algorithm, returns mean and precision for each player
mean_player_skills, precision_player_skills, mcov2, pcov2 = eprank(G, M, num_iters)
#print(pcov)
print(mcov2[:,0])