import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal 

##############################################################################################
#
#   Author: Ani Bhalekar
#   Date: 27/07/2024 
#   
#   Purpose: 
#
#           Application of EM algorithm with a GMM class. Look at main sample code
#           for example usage. EM is sensitive to initialisation to initial conditions
#           model collapse can occur due to bad initialisation - change init_mu and 
#           init_sigma at your own risk
#
##############################################################################################
class GMM():
    def __init__(self, k, dim, init_mu=None, init_sigma=None, init_pi=None, colors=None):
        
        self.k = k
        self.dim = dim
        if(init_mu is None):
            init_mu = (np.random.rand(k, dim) - 0.5)
        self.mu = init_mu 
        if(init_sigma is None):
            init_sigma = np.zeros((k, dim, dim))
            for i in range(k):
                init_sigma[i] = np.eye(dim) * 10
        self.sigma = init_sigma
        if(init_pi is None):
            init_pi = np.ones(self.k)/self.k
        self.pi = init_pi
        if(colors is None):
            colors = np.random.rand(k, 3)
        self.colors = colors
    
    def init_em(self, X):
        ########################################################################################
        #Initialization for EM algorithm.
        #input:
        #    - X: data (batch_size, dim)
        ########################################################################################
        self.data = X
        self.num_points = X.shape[0]
        self.z = np.zeros((self.num_points, self.k))
    
    def e_step(self):
        #######################################################################################
        #E-step of EM algorithm.
        #######################################################################################
        for i in range(self.k):
            
            self.z[:, i] = self.pi[i] * multivariate_normal.pdf(self.data, mean=self.mu[i], cov=self.sigma[i], allow_singular=True)
            
        self.z /= self.z.sum(axis=1, keepdims=True)
    
    def m_step(self):
        #######################################################################################
        #M-step of EM algorithm.
        #######################################################################################
        sum_z = self.z.sum(axis=0)
        self.pi = sum_z / self.num_points
        self.mu = np.matmul(self.z.T, self.data)
        self.mu /= sum_z[:, None]
        for i in range(self.k):
            j = np.expand_dims(self.data, axis=1) - self.mu[i]
            s = np.matmul(j.transpose([0, 2, 1]), j)
            self.sigma[i] = np.matmul(s.transpose(1, 2, 0), self.z[:, i] )
            self.sigma[i] /= sum_z[i]
        
        self.sigma += np.ones_like(self.sigma) * 0.1
    

