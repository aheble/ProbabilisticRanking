# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 16:19:01 2021

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
# number of iterations
num_iters = 3000
iters_axis = [a for a in range(50,num_iters,10)]
# perform gibbs sampling, skill samples is an num_players x num_samples array
skill_samples,mcov,pcov = gibbs_sample(G, M, num_iters) # random_nums
plt.figure(figsize=[12,3])
plt.plot(skill_samples[0][50::10],color="k")
plt.xlabel("Sample")
plt.ylabel("Player 1 Skill Rating")
plt.show()
plt.figure(figsize=[12,3])
plt.plot(skill_samples[49][50::10],color="r")
plt.xlabel("Saample")
plt.ylabel("Player 50 Skill Rating")
plt.show()
plt.figure(figsize=[12,3])
plt.plot(skill_samples[94][50::10],color="c")
plt.xlabel("Sample")
plt.ylabel("Player 95 Skill Rating")
plt.show()
# Code for plotting the autocorrelation function for player p
plt.figure(figsize=[10,4])
for p in range(M):
    autocor = np.zeros(15)
    for i in range(15):
        autocor[i]=pandas.Series.autocorr(pandas.Series(skill_samples[p,:][50::10]),lag=i)
    plt.plot(abs(autocor))
    
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.show()

