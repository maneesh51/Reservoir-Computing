# -*- coding: utf-8 -*-
"""
Created on Mon May 15 13:18:21 2023

@author: Manish Yadav
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplot

import networkx as nx
from numpy.linalg import inv
import os
import random
from tqdm.notebook import tqdm, trange

#### Generate Reservoir Matrix
def ResMat(N, ConnProb, Spectral_radius):
    G = nx.erdos_renyi_graph(N, ConnProb, seed=None, directed=True)
    
    ### Remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    N = G.number_of_nodes()
    
    ### Randomize weights between (0, 1)
    Rand_weights = np.random.random((N,N))
    GNet = nx.to_numpy_matrix(G)
    GNet = np.multiply(GNet,Rand_weights)
    
    ### Rescaling to a desired spectral radius 
    Spectral_radius_GNet = max(abs(np.linalg.eigvals(GNet)))
    ResMat = GNet*Spectral_radius/Spectral_radius_GNet
    
    return G, ResMat

#### Reservoir Computer
def Reservoir(GNet, Init, U, Winp, N, alpha):
    Nodes_res = GNet.shape[0]; 
    Npts_U = len(U)

    R = np.zeros([N, Npts_U])
    R[:,0] = Init
    
    #### time loop
    for t in range(0, Npts_U-1):      
        R[:,t+1] = (1 - alpha)*np.asarray(R[:,t]) + alpha*np.tanh(np.dot(GNet, R[:,t].T) + Winp*U[t] )
        
    return R

#### Ridge-Regression
def Ridge_Regression(N, R, beta, V_train):
    W_out = np.dot(np.dot(V_train, R.T), np.linalg.inv((np.dot(R, R.T) + beta*np.identity(N))))
    return W_out

#### Errors
def Errors(y_predicted, y_actual):
    MSE = np.mean(np.square(np.subtract(y_predicted,y_actual)))
    Variance = (np.mean(np.square(np.subtract(y_actual, np.mean(y_actual) ) ) ) )
    NMSE = MSE/Variance
    NRMSE = np.sqrt(NMSE)
    return NMSE, NRMSE 

