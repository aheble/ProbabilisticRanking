import scipy.linalg
import numpy as np
from tqdm import tqdm
def gibbs_sample(G, M, num_iters):
    # number of games
    N = G.shape[0]
    # Array containing mean skills of each player, set to prior mean
    w = np.zeros((M, 1))
    # Array that will contain skill samples
    skill_samples = np.zeros((M, num_iters))
    # Array containing skill variance for each player, set to prior variance
    pv = 0.5 * np.ones(M)
    # Initialise mean and variance trackers
    mcov=np.zeros((num_iters,M))
    pcov=np.zeros((num_iters,M))
    
    # number of iterations of Gibbs
    for i in tqdm(range(num_iters)):
        # sample performance given differences in skills and outcomes
        t = np.zeros((N, 1))
        for g in range(N):

            s = w[G[g, 0]] - w[G[g, 1]]  # difference in skills
            t[g] = s + np.random.randn()  # Sample performance
            while t[g] < 0:  # rejection step
                t[g] = s + np.random.randn()  # resample if rejected

        # Jointly sample skills given performance differences
        m = np.zeros((M, 1))
        #print("We are here, doing mean matrix, iteration {}".format(i))
        for ind,(p1,p2) in enumerate(G): m[p1]+=t[ind]; m[p2]-=t[ind]
        iS = np.zeros((M, M))  # Container for sum of precision matrices (likelihood terms)
        #print("We are here, doing iS matrix, iteration {}".format(i))
        for g in range(N):
            iS[G[g,0],G[g,0]]+=1; iS[G[g,1],G[g,1]]+=1; iS[G[g,0],G[g,1]]-=1; iS[G[g,1],G[g,0]]-=1
            #print("Current iteration for sums: {}".format(g))
            
        # Posterior precision matrix
        iSS = iS + np.diag(1. / pv)
        
        # Use Cholesky decomposition to sample from a multivariate Gaussian
        iR = scipy.linalg.cho_factor(iSS)  # Cholesky decomposition of the posterior precision matrix
        mu = scipy.linalg.cho_solve(iR, m, check_finite=False)  # uses cholesky factor to compute inv(iSS) @ m
        
        # Update trackers
        pcov[i] = np.diag(np.linalg.inv(iSS))
        mcov[i] = mu.flat

        # sample from N(mu, inv(iSS))
        w = mu + scipy.linalg.solve_triangular(iR[0], np.random.randn(M, 1), check_finite=False)
        skill_samples[:, i] = w[:, 0]
        
    return skill_samples, mcov, pcov