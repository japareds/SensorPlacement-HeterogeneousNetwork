#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:01:55 2023

@author: jparedes
"""
import numpy as np
from scipy import linalg
from sklearn.metrics import mean_absolute_error,mean_squared_error
import Formulas
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
            if len(loc_eps) !=0:
                C_eps = In[loc_eps,:]
            else:
                C_eps = []
            if len(loc_zero) !=0:
                C_zero = In[loc_zero,:]
            else:
                C_zero = []
            if len(loc_empty)!=0:
                C_empty = In[loc_empty,:]
            else:
                C_empty = []
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
        
    def Cov_matrix_limit(self,Psi):
        """
        Compute covariance matrix in the limit var_zero = 0

        Parameters
        ----------
        Psi : numpy array
            low-rank basis

        Returns
        -------
        None.

        """
        self.Covariances = {el:0 for el in np.arange(len(self.locations))}
        for idx in self.locations.keys():
            C_eps = self.C[idx][0]
            C_zero = self.C[idx][1]
            
            
            Theta_eps = C_eps@Psi
            Theta_zero = C_zero@Psi
            
            refst_matrix = Theta_zero.T@Theta_zero
            refst_pinv = np.linalg.pinv(refst_matrix)
            
            lcs_matrix = Theta_eps.T@Theta_eps
            lcs_pinv = np.linalg.pinv(lcs_matrix)
            
            
            Is = np.identity(self.r)
            term = Is - refst_matrix@refst_pinv
            self.Covariances[idx] = term@lcs_pinv@term
        
    
    
    def perturbate_signal(self,ds_signal,variance,seed):
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0,scale=variance,size=ds_signal.shape)
        return ds_signal+noise
    
    def covariance_matrix_GLS(self,Psi,var_eps,var_zero):
        """
        Compute GLS covariance matrix

        Parameters
        ----------
        Psi : numpy array
            low-rank basis
        var_eps : float
            LCS variance
        var_zero : float
            Ref st. variance
        
        Returns
        -------
        None.

        """
        Covariances = {el:0 for el in np.arange(len(self.locations))}
       
        for idx in self.locations.keys():
            C_eps = self.C[idx][0]
            C_zero = self.C[idx][1]
            Theta_eps = C_eps@Psi
            Theta_zero = C_zero@Psi
            
            if C_eps.shape[0] == 0:# no LCS
                Precision_matrix = (self.var_zero**-1)*Theta_zero.T@Theta_zero
            elif C_zero.shape[0] == 0:#no Ref.St.
                Precision_matrix = (self.var_eps**-1)*Theta_eps.T@Theta_eps
            else:
                Precision_matrix = (var_eps**(-1)*Theta_eps.T@Theta_eps) + (var_zero**(-1)*Theta_zero.T@Theta_zero)
                
            S = np.linalg.svd(Precision_matrix)[1]
            rcond_pinv = rcond_pinv = (S[-1]+S[-2])/(2*S[0])
            Covariances[idx] = np.linalg.pinv(Precision_matrix,rcond_pinv)
            
        self.Covariances = Covariances
        
    def covariance_matrix_limit(self,Psi,var_eps,r):
        """
        Compute covariance matrix in the limit var_zero == 0

        Parameters
        ----------
        Psi : numpy array
            low-rank basis
        var_eps : float
            LCS variance
        
        
        Returns
        -------
        None.

        """
        Covariances = {el:0 for el in np.arange(len(self.locations))}
       
        for idx in self.locations.keys():
            C_eps = self.C[idx][0]
            C_zero = self.C[idx][1]
            C_empty = self.C[idx][2]
            
            
            if self.p_eps == 0:#no LCSs
                Cov = np.zeros(shape=(r,r))
            elif self.p_zero == 0:#no RefSt
                Theta_eps = C_eps@Psi
                Precision_matrix = (var_eps**-1)*Theta_eps.T@Theta_eps
                S = np.linalg.svd(Precision_matrix)[1]
                rcond_pinv = rcond_pinv = (S[-1]+S[-2])/(2*S[0])
                Cov = np.linalg.pinv( Precision_matrix,rcond_pinv)
            else: # compute covariance matrix using projector
                Theta_eps = C_eps@Psi
                Theta_zero = C_zero@Psi
                
                
                refst_matrix = Theta_zero.T@Theta_zero
                Is = np.identity(r)
                try:
                    P = Is - refst_matrix@np.linalg.pinv(refst_matrix)
                except:
                    P = Is - refst_matrix@np.linalg.pinv(refst_matrix,hermitian=True,rcond=1e-10)
                
                
                rank1 = np.linalg.matrix_rank(Theta_eps@P,tol=1e-10)
                rank2 = np.linalg.matrix_rank(P@Theta_eps.T,tol=1e-10)
                
                S1 = np.linalg.svd(Theta_eps@P)[1]
                S2 = np.linalg.svd(P@Theta_eps.T)[1]
                
                if rank1==min((Theta_eps@P).shape):
                    try:
                        rcond1_pinv = (S1[-1]+S1[-2])/(2*S1[0])
                    except:
                        rcond1_pinv = 1e-15
                else:
                    rcond1_pinv = (S1[rank1]+S1[rank1-1])/(2*S1[0])
                
                if rank2==min((P@Theta_eps.T).shape):
                    try:
                        rcond2_pinv = (S2[-1]+S2[-2])/(2*S2[0])
                    except:
                        rcond2_pinv = 1e-15
                else:
                    rcond2_pinv = (S2[rank2]+S2[rank2-1])/(2*S2[0])
                  
                    
                    
                    
                Cov = var_eps*np.linalg.pinv(Theta_eps@P,rcond=rcond1_pinv)@np.linalg.pinv(P@Theta_eps.T,rcond=rcond2_pinv)
                  
                
            Covariances[idx] = Cov
        
            
        self.Covariances = Covariances
        
    
    def beta_estimated_GLS(self,Psi,ds_lcs,ds_refst,var_eps,var_zero):
        """
        Compute GLS estimated regressor

        Parameters
        ----------
        Psi : np array
            low-rank basis
        ds_lcs : pandas dataframe
            LCSs dataset
        ds_refst : pandas dataframe
            reference stations dataset
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
            
            y_refst = ds_lcs.loc[:,locations[1]]
            y_lcs = ds_refst.loc[:,locations[0]]
            
            second_term = (var_zero**-1)*Theta_zero.T@y_refst.T + (var_eps**-1)*Theta_eps.T@y_lcs.T
            self.beta_hat[idx] = Cov@second_term
            
    def beta_estimated_limit(self,Psi,ds_lcs,ds_refst,r):
        """
        Compute estimated regressor (beta) from sensor measurements
        in the limit variances refst goes to zero (limit of GLS)

        Parameters
        ----------
        Psi : numpy array
            sparse basis
        ds_lcs : pandas dataframe
           LCSs dataset
        ds_refst : pandas dataframe
           reference stations dataset

        Returns
        -------
        self.beta_hat : numpy array
                estimated regressor over time (r,num_samples)

        """
        
        
        self.beta_hat = {el:0 for el in np.arange(len(self.locations))}
        
        for idx in self.locations.keys():
            C_lcs = self.C[idx][0]
            C_refst = self.C[idx][1]
            
            Theta_lcs = C_lcs@Psi
            Theta_refst = C_refst@Psi
            
            refst_matrix = Theta_refst.T@Theta_refst
            
            Is = np.identity(r)
            P = Is - refst_matrix@np.linalg.pinv(refst_matrix)
            
            term_refst = np.linalg.pinv(Theta_refst) #np.linalg.pinv(refst_matrix)@Theta_refst.T@y_refst
            term_lcs = np.linalg.pinv(Theta_lcs@P)@np.linalg.pinv(P@Theta_lcs.T)@Theta_lcs.T
            
            locations = self.locations[idx]
            y_refst = ds_lcs.loc[:,locations[1]].T
            y_lcs = ds_refst.loc[:,locations[0]].T
            
            
            self.beta_hat[idx] = term_lcs@y_lcs + term_refst@y_refst 
            
            
        
            
    def estimation(self,Psi,ds_real):
        self.rmse_full = {el:0 for el in np.arange(len(self.locations))}
        self.rmse_unmonitored = {el:0 for el in np.arange(len(self.locations))}
        self.rmse_refst = {el:0 for el in np.arange(len(self.locations))}
        self.rmse_lcs = {el:0 for el in np.arange(len(self.locations))}
       
        for idx in self.locations.keys():
            locations = self.locations[idx]
            y_real = ds_real
            
            # estimation whole network
            y_hat = Psi@self.beta_hat[idx]
            y_pred = y_hat.T
            y_pred.columns = y_real.columns
            
            # estimate error in whole network
            #self.rmse_full[idx] = np.sqrt(mean_squared_error(y_real, y_pred))/self.n
            self.rmse_full[idx] = np.median([np.sqrt(mean_squared_error(y_real.loc[:,i],y_pred.loc[:,i])) for i in y_real.columns])
            
            
            # compute error in unmonitored locations
            #self.rmse_unmonitored[idx] = np.sqrt(mean_squared_error(y_real.loc[:,locations[2]],y_pred.loc[:,locations[2]]))/self.p_empty
            self.rmse_unmonitored[idx] = np.median([np.sqrt(mean_squared_error(y_real.loc[:,locations[2]].loc[:,i],y_pred.loc[:,locations[2]].loc[:,i])) for i in locations[2]])
            
            
            # estimate error in ref st locations
            #self.rmse_refst[idx] = np.sqrt(mean_squared_error(y_real.loc[:,locations[1]],y_pred.loc[:,locations[1]]))/self.p_zero
            self.rmse_refst[idx] = np.median([np.sqrt(mean_squared_error(y_real.loc[:,locations[1]].loc[:,i],y_pred.loc[:,locations[1]].loc[:,i])) for i in locations[1]])
            
            # estimate error in LCSs locations
            #self.rmse_lcs[idx] = np.sqrt(mean_squared_error(y_real.loc[:,locations[0]],y_pred.loc[:,locations[0]]))/self.p_eps
            self.rmse_lcs[idx] = np.median([np.sqrt(mean_squared_error(y_real.loc[:,locations[0]].loc[:,i],y_pred.loc[:,locations[0]].loc[:,i])) for i in locations[0]])
            
            
                        
            
            
            
            
            
            
            
            
            
        
        
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
    
    
    
    