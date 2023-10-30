#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:01:55 2023

@author: jparedes
"""
import numpy as np
from scipy import linalg
from sklearn.metrics import mean_absolute_error,mean_squared_error
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
        
        if self.p_eps != 0:
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
        else:
            loc_eps = []
            for i in np.arange(num_samples):
                rng.shuffle(locations)
                loc_zero = np.sort(locations[:self.p_zero])
                if self.p_empty!=0:
                    loc_empty = np.sort(locations[self.p_zero:])
                else:
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
    
    def Cov_metric(self,Psi,sigma_eps,sigma_zero,criteria='D_optimal',compute_empty=False):
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
        compute_empty : bool
            compute covariance matrix on non-monitored locations rather than on monitored ones
            
        Returns
        -------
        self.metric: dict

        """
        metric = {el:0 for el in np.arange(len(self.locations))}
        metric_precisionMat = {el:0 for el in np.arange(len(self.locations))}
        Covariances = {el:0 for el in np.arange(len(self.locations))}
        
        if criteria not in ['logdet','eigval','WCS']:
            print(f'Chosen criteria {criteria} is not valid.')
            return
        
        for idx in self.locations.keys():
            C_eps = self.C[idx][0]
            C_zero = self.C[idx][1]
            C_empty = self.C[idx][2]
            
            
            Theta_eps = C_eps@Psi
            Theta_zero = C_zero@Psi
            Theta_empty = C_empty@Psi
            if compute_empty and self.p_empty!=0:
                Precision_matrix = Theta_empty.T@Theta_empty
                try:
                    Cov = np.linalg.inv( Precision_matrix ) 
                except:
                    print(f'Computing pseudo-inverse for index {idx}')
                    Cov = np.linalg.pinv( Precision_matrix )
                
            else:
            
                if self.p_eps !=0:
                    Precision_matrix = (sigma_eps**(-1)*Theta_eps.T@Theta_eps) + (sigma_zero**(-1)*Theta_zero.T@Theta_zero)
                    try:
                        Cov = np.linalg.inv( Precision_matrix ) 
                    except:
                        print(f'Computing pseudo-inverse for index {idx}')
                        Cov = np.linalg.pinv( Precision_matrix )
                else:
                    Precision_matrix = sigma_zero**(-1)*Theta_zero.T@Theta_zero
                    try:
                        Cov = np.linalg.inv(Precision_matrix)
                    except:
                        print(f'Computing pseudo-inverse for index {idx}')
                        Cov = np.linalg.pinv( Precision_matrix)
            
            Covariances[idx] = Cov
            if criteria == 'logdet':
                metric[idx] = np.log(np.linalg.det(Cov))
                metric_precisionMat[idx] = np.log(np.linalg.det(Precision_matrix))
            elif criteria == 'eigval':
                metric[idx] = np.max(np.real(np.linalg.eig(Cov)[0]))
                metric_precisionMat[idx] = np.max(np.real(np.linalg.eig(Precision_matrix)[0]))
                
            elif criteria == 'WCS':
                metric[idx] = np.diag(Cov).max()
        
        self.criteria = criteria
        self.metric = metric
        self.metric_precisionMat = metric_precisionMat
        self.Covariances = Covariances
    
    def perturbate_signal(self,ds_signal,variance,seed):
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0,scale=variance,size=ds_signal.shape)
        return ds_signal+noise
    
    def beta_estimated(self,Psi,ds_estimation,var_eps,var_zero):
        """
        Compute GLS estimated regressor

        Parameters
        ----------
        Psi : np array
            low-rank basis
        ds_estimation : pandas dataframe
            dataset for estimation
        var_eps : float
            LCSs variance
        var_zero : float
            Ref.St. variance

        Returns
        -------
        None.

        """
        self.beta_hat = {el:0 for el in np.arange(len(self.locations))}
        for idx in self.locations.keys():
            C_eps = self.C[idx][0]
            C_zero = self.C[idx][1]
            
            Theta_eps = C_eps@Psi
            Theta_zero = C_zero@Psi
            
            Cov = self.Covariances[idx]
            
            locations = self.locations[idx]
            
            y_refst = self.perturbate_signal(ds_estimation.loc[:,locations[1]], var_zero*15, seed=idx)
            y_lcs = self.perturbate_signal(ds_estimation.loc[:,locations[0]], var_eps*15, seed=idx)
            #y_refst = ds_estimation.loc[:,locations[1]]
            #y_lcs = ds_estimation.loc[:,locations[0]]
            
            second_term = (var_zero**-1)*Theta_zero.T@y_refst.T + (var_eps**-1)*Theta_eps.T@y_lcs.T
            self.beta_hat[idx] = Cov@second_term
            
    def estimation(self,Psi,ds_estimation,var_eps,var_zero,locations_to_estimate='empty'):
        self.rmse = {el:0 for el in np.arange(len(self.locations))}
        self.mae = {el:0 for el in np.arange(len(self.locations))}
       
        for idx in self.locations.keys():
            locations = self.locations[idx]
            if locations_to_estimate == 'empty':
                l = 2
                y_estimation = ds_estimation.loc[:,locations[l]]
                C_estimation = self.C[idx][l]
            elif locations_to_estimate == 'LCSs':
                l = 0
                var_noise = var_eps
                y_estimation = self.perturbate_signal(ds_estimation.loc[:,locations[l]], var_noise*15, seed=idx)
                C_estimation = self.C[idx][l]
            elif locations_to_estimate == 'RefSt':
                l = 1
                var_noise = var_zero
                y_estimation = self.perturbate_signal(ds_estimation.loc[:,locations[l]], var_noise*15, seed=idx)
                C_estimation = self.C[idx][l]
            else:
                y_estimation = ds_estimation
                C_estimation = np.identity(ds_estimation.shape[1])
                
            
            # format tabular dataset
            y_hat = C_estimation@Psi@self.beta_hat[idx]
            y_pred = y_hat.T
            y_pred.columns = y_estimation.columns
                        
            self.rmse[idx] = np.sqrt(mean_squared_error(y_estimation, y_pred))
            self.mae[idx] = mean_absolute_error(y_estimation, y_pred)
            
            
            
            
            
            
        
        
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
    
    
    
    