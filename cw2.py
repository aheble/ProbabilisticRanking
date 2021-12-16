import matplotlib.pyplot as plt
import numpy as np


def sorted_barplot(P, W):
    """
    Function for making a sorted bar plot based on values in P, and labelling the plot with the
    corresponding names
    :param P: An array of length num_players (107)
    :param W: Array containing names of each player
    :return: None
    """
    M = len(P)
    Pplot = []
    xx = np.linspace(0, M, M)
    plt.figure(figsize=(20, 40))
    sorted_indices = np.argsort(P)
    print(sorted_indices)
    for i in sorted_indices:
        Pplot.append(P[i])
    sorted_names = W[sorted_indices]
    #print(sorted_names)
    plt.title("Gibbs Sampling")
    plt.barh(xx, Pplot,color="r")
    plt.yticks(np.linspace(0, M, M), labels=sorted_names[:, 0])
    plt.ylim([-2, 109])
    plt.xlabel("Average Probability of Winning")
    plt.ylabel("Player")
    plt.show()

