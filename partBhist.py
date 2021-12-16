# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 15:51:42 2021

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

def calc(m,v,burn,fin):
    return (sum(m[burn:fin])/len(m[burn:fin]), sum(v[burn:fin])/len(v[burn:fin]))


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
skill_samples, m, p = gibbs_sample(G, M, num_iters) # random_nums
std = [math.sqrt(a) for a in p[:,0]]
# for plotting
x=np.linspace(1.0,2.0,200)

plt.figure(figsize=[8,3])
plt.hist(x=m[:,0][50:150:10],bins=25,density=True)
plt.plot(x,stats.norm.pdf(x,calc(m[:,0],std,50,150)[0],calc(m[:,0],std,50,150)[1]))
plt.xlabel("Skill Rating")
plt.ylabel("Density")
plt.title("10 Samples")
plt.text(1.8,2,"mean: {}\nvariance: {}".format(round(calc(m[:,0],std,50,150)[0],2),round(calc(m[:,0],std,50,150)[1]**2,2)))
plt.show()

plt.figure(figsize=[8,3])
plt.hist(x=m[:,0][50:550:10],bins=25,density=True)
plt.plot(x,stats.norm.pdf(x,calc(m[:,0],std,50,550)[0],calc(m[:,0],std,50,550)[1]))
plt.xlabel("Skill Rating")
plt.ylabel("Density")
plt.title("50 Samples")
plt.text(1.8,2.5,"mean: {}\nvariance: {}".format(round(calc(m[:,0],std,50,550)[0],2),round(calc(m[:,0],std,50,550)[1]**2,2)))
plt.show()

plt.figure(figsize=[8,3])
plt.hist(x=m[:,0][50:1050:10],bins=25,density=True)
plt.plot(x,stats.norm.pdf(x,calc(m[:,0],std,50,1050)[0],calc(m[:,0],std,50,1050)[1]))
plt.xlabel("Skill Rating")
plt.ylabel("Density")
plt.title("100 Samples")
plt.text(1.8,2.5,"mean: {}\nvariance: {}".format(round(calc(m[:,0],std,50,1050)[0],2),round(calc(m[:,0],std,50,1050)[1]**2,2)))
plt.show()

plt.figure(figsize=[8,3])
plt.hist(x=m[:,0][50:2550:10],bins=25,density=True)
plt.plot(x,stats.norm.pdf(x,calc(m[:,0],std,50,1050)[0],calc(m[:,0],std,50,1050)[1]))
plt.xlabel("Skill Rating")
plt.ylabel("Density")
plt.title("250 Samples")
plt.text(1.8,2.5,"mean: {}\nvariance: {}".format(round(calc(m[:,0],std,50,1050)[0],2),round(calc(m[:,0],std,50,1050)[1]**2,2)))
plt.show()