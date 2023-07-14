#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:01:55 2023

@author: jparedes
"""
import os
import numpy as np
from scipy import linalg
# ===================================================================================
# Random sensor placement: generate random placement locations for (LCSs,RefSt,Empty)
# ===================================================================================
class randomPlacement():
    def __init__(self,p_eps,p_zero,p_empty,n):
        """
        Network parameters

        Parameters
        ----------
        p_eps : int
            number of low-cost sensors
        p_zero : int
            number of reference stations
        p_empty : int
            number of empty (unmonitored) locations
        n : int
            total number of locations in the network

        Returns
        -------
        None.

        """
        self.p_eps = p_eps
        self.p_zero = p_zero
        self.p_empty = p_empty
        self.n = n
        
    
    def place_sensors(self,num_samples=10,random_seed = 92):
        """
        Generates index for LCSs, refSt, Empty locations in the network

        Parameters
        ----------
        num_samples : int, optional
            Number of iterations for random placement. The default is 10.
        random_seed : int, optional
            Random seed used for pseudo-rng used to place sensors

        Returns
        -------
        self.locations: dict
            Each entry is an iteration of random placement. 
            For a given entry, it contains indexes for each type of sensor

        """
        locations = np.arange(self.n)
        random_locations = {el:0 for el in np.arange(num_samples)}
        rng = np.random.default_rng(seed=random_seed)
        for i in np.arange(num_samples):
            rng.shuffle(locations)
            loc_eps = np.sort(locations[:self.p_eps])
            if self.p_empty != 0:
                loc_zero = np.sort(locations[self.p_eps:-self.p_empty])
                loc_empty = np.sort(locations[-self.p_empty:])
            else:
                loc_zero = np.sort(locations[self.p_eps:])
                loc_empty = []
            random_locations[i] = [loc_eps,loc_zero,loc_empty]
        
        # random_locations = [idx(LCSs),idx(RefSt),idx(Empty)]
        self.locations = random_locations
        
    def C_matrix(self):
        """
        Convert indexes of LCSs, RefSt and Emtpy locations into
        C matrix

        Returns
        -------
        self.C: dict
            Each entry is an iteration of random placement
            For a given entry, it contains a list of C matrix for LCSs, RefSt, Empty
            

        """
        In = np.identity(self.n)
        C = {el:0 for el in np.arange(len(self.locations))}
        for idx in self.locations.keys():
            loc_eps = self.locations[idx][0]
            loc_zero = self.locations[idx][1]
            loc_empty = self.locations[idx][2]
            C_eps = In[loc_eps,:]
            C_zero = In[loc_zero,:]
            C_empty = In[loc_empty,:]
            C[idx] = [C_eps,C_zero,C_empty]
        self.C = C
    
    def design_metric(self,Psi,sigma_eps,sigma_zero,criteria='D_optimal'):
        """
        Computes experiment design performance metric
        
        Parameters
        ----------
        Psi: list
            nxr low-rank basis of network signal
        sigma_eps: float
            variance of LCSs
        sigma_zero: float
            variance of RefSt
        criteria: str
            experiemtn design criteria used to compute metric
            
        Returns
        -------
        self.metric: dict

        """
        metric = {el:0 for el in np.arange(len(self.locations))}
        
        for idx in self.locations.keys():
            C_eps = self.C[idx][0]
            C_zero = self.C[idx][1]
            C_empty = self.C[idx][2]
            
            Theta_eps = C_eps@Psi
            Theta_zero = C_zero@Psi
            try:
                Cov = np.linalg.inv( (sigma_eps**(-1)*Theta_eps.T@Theta_eps) + (sigma_zero**(-1)*Theta_zero.T@Theta_zero) ) 
            except:
                print(f'Computing pseudo-inverse for index {idx}')
                Cov = np.linalg.pinv( (sigma_eps**(-1)*Theta_eps.T@Theta_eps) + (sigma_zero**(-1)*Theta_zero.T@Theta_zero) )
            if criteria == 'D_optimal':
                metric[idx] = np.log(np.linalg.det(Cov))
            elif criteria == 'E_optimal':
                metric[idx] = np.max(np.real(np.linalg.eig(Cov)[0]))
        
        self.criteria = criteria
        self.metric = metric
        
if __name__=='__main__':
    print('Testing')
    p_eps,p_zero,p_empty,n,r = 2,3,1,6,3
    sigma_eps,sigma_zero = 1,1e-2
    rng = np.random.default_rng(seed=40)
    U = linalg.orth(rng.random((n,n)))
    Psi = U[:,:r]
    random_placement = randomPlacement(p_eps,p_zero,p_empty,n)
    random_placement.place_sensors(num_samples=10,random_seed=92)
    random_placement.C_matrix()
    random_placement.design_metric(Psi, sigma_eps, sigma_zero)
    
    
    
    