#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:14:31 2023

@author: jparedes
"""


import os
import numpy as np
import scipy.stats
import pickle
from sklearn.metrics import mean_absolute_error,mean_squared_error
import DataSet as DS
import LowRankBasis as LRB
import randomPlacement as RP
import SensorPlacement as SP
import Plots


# =============================================================================
# Estimate sensor measurements at unmonitored locations
# =============================================================================



class Estimation():
    def __init__(self,n,r,p_empty,p_zero_estimate,var_eps,var_zero,num_random_placements,alpha_reg,ds_lcs,ds_refst,ds_real,Psi,Dopt_path,rank_path):
        # network parameters
        self.n = n
        self.r = r
        self.p_empty =p_empty
        self.p_zero_estimate = p_zero_estimate
        self.p_eps_estimate = self.n-(self.p_zero_estimate+self.p_empty)
        # variances
        self.var_eps = var_eps
        self.var_zero = var_zero
        self.num_random_placements = num_random_placements
        self.alpha_reg = alpha_reg
        # dataset
        self.ds_lcs = ds_lcs
        self.ds_refst = ds_refst
        self.ds_real = ds_real
        self.Psi = Psi
        # files path
        self.Dopt_path = Dopt_path
        self.rank_path = rank_path
        
        
        
    
    def mean_confidence_interval(self,data, confidence=0.90):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
        return np.array([m, m-h, m+h])
        
        
        
    
    def perturbate_signal(self,ds_signal,variance,seed):
        """
        Add noise to signal

        Parameters
        ----------
        ds_signal : pandas dataframe
            original signal. Multiple measurements over time
        variance : float
            noise variance
        seed : int
            rng seed

        Returns
        -------
        pandas dataframe
            perturbated signal
        """
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0,scale=variance,size=ds_signal.shape)
        return ds_signal+noise
    
    
    def get_estimated_measurements(self,beta_hat,y_real):
        
        # predicted values in all network
        y_hat = self.Psi@beta_hat
        y_pred = y_hat.T
        y_pred.columns = y_real.columns
        
        return y_pred
    
    def compute_metrics(self,y_real,y_pred,y_refst,y_lcs,y_empty):
        """
        Compute estimation error

        Parameters
        ----------
        y_real : pandas dataframe
            Full entwork actual values
        y_pred : pandas dataframe
            Full network predicted values
        y_refst : pandas dataframe
            Reference stations measurements at locations
        y_lcs : pandas dataframe
            LCSs measurements at locations
        y_empty : pandas dataframe
            True values at unmonitored locations
    
        Returns
        -------
        [rmse_full,rmse_refst,rmse_lcs,rmse_unmonitored]
            RMSE at different locations
        """
        loc_refst = [i for i in y_refst.columns]
        loc_lcs = [i for i in y_lcs.columns]
        loc_unmonitored = [i for i in y_empty.columns]
        
        
        
        # full network metric
        
        #rmse_full = np.sqrt(mean_squared_error(y_real, y_pred))/self.n
        rmse_full = np.median([np.sqrt(mean_squared_error(y_real.loc[:,i],y_pred.loc[:,i])) for i in y_real.columns])
        
        # error at reference stations locations
        
        #rmse_refst = np.sqrt(mean_squared_error(y_real.loc[:,y_refst.columns],y_pred.loc[:,y_refst.columns]))/self.p_zero_estimate
        rmse_refst = np.median([np.sqrt(mean_squared_error(y_real.loc[:,loc_refst].loc[:,i],y_pred.loc[:,loc_refst].loc[:,i])) for i in loc_refst])
            
        # error at LCSs locations
        #rmse_lcs = np.sqrt(mean_squared_error(y_real.loc[:,y_lcs.columns], y_pred.loc[:,y_lcs.columns]))/self.p_eps_estimate
        rmse_lcs = np.median([np.sqrt(mean_squared_error(y_real.loc[:,loc_lcs].loc[:,i],y_pred.loc[:,loc_lcs].loc[:,i])) for i in loc_lcs])
        
        #rmse_unmonitored = np.sqrt(mean_squared_error(y_real.loc[:,y_empty.columns], y_pred.loc[:,y_empty.columns]))/self.p_empty
        rmse_unmonitored = np.median([np.sqrt(mean_squared_error(y_real.loc[:,loc_unmonitored].loc[:,i],y_pred.loc[:,loc_unmonitored].loc[:,i])) for i in loc_unmonitored])
            
    
        return [rmse_full,rmse_refst,rmse_lcs,rmse_unmonitored]
        
    
    def random_placement_estimation(self, random_path):
        
        fname = random_path+f'randomPlacement_locations_r{self.r}_pEmpty{self.p_empty}_numRandomPlacements{self.num_random_placements}.pkl'
        with open(fname,'rb') as f:
            dict_random_locations = pickle.load(f)
            
        
        
        # random placement estimation
        random_placement = RP.randomPlacement(self.p_eps_estimate,self.p_zero_estimate,self.p_empty,self.n)
        # obtain locations and covariance matrices
        random_placement.locations = dict_random_locations[self.p_zero_estimate]
        random_placement.C_matrix()
        random_placement.covariance_matrix_GLS(self.Psi, self.var_eps, self.var_zero)
        
        # get estimated measurements using beta GLS
        random_placement.beta_estimated_GLS(self.Psi,self.ds_lcs,self.ds_refst,self.var_eps,self.var_zero)
        random_placement.estimation(self.Psi,self.ds_real)
        
        rmse_random_full = np.array([i for i in random_placement.rmse_full.values()])
        rmse_random_refst = np.array([i for i in random_placement.rmse_refst.values()])
        rmse_random_lcs = np.array([i for i in random_placement.rmse_lcs.values()])
        rmse_random_unmonitored = np.array([i for i in random_placement.rmse_unmonitored.values()])
        
        self.rmse_random_full = rmse_random_full.mean()
        self.rmse_random_refst = rmse_random_refst.mean()
        self.rmse_random_lcs = rmse_random_lcs.mean()
        self.rmse_random_unmonitored = rmse_random_unmonitored.mean()
        
      
    def random_placement_estimation_limit(self, random_path):
        
        fname = random_path+f'randomPlacement_locations_r{self.r}_pEmpty{self.p_empty}_numRandomPlacements{self.num_random_placements}.pkl'
        with open(fname,'rb') as f:
            dict_random_locations = pickle.load(f)
            
       
        
        # random placement estimation
        random_placement = RP.randomPlacement(self.p_eps_estimate,self.p_zero_estimate,self.p_empty,self.n)
        # obtain locations
        random_placement.locations = dict_random_locations[self.p_zero_estimate]
        random_placement.C_matrix()
        
        # get estimated measurements using beta(epsilon=0)
        random_placement.beta_estimated_limit(self.Psi,self.ds_lcs,self.ds_refst,self.r)
        random_placement.estimation(self.Psi,self.ds_real)
        
        rmse_random_full = np.array([i for i in random_placement.rmse_full.values()])
        rmse_random_refst = np.array([i for i in random_placement.rmse_refst.values()])
        rmse_random_lcs = np.array([i for i in random_placement.rmse_lcs.values()])
        rmse_random_unmonitored = np.array([i for i in random_placement.rmse_unmonitored.values()])
        
        self.rmse_random_full = rmse_random_full.mean()
        self.rmse_random_refst = rmse_random_refst.mean()
        self.rmse_random_lcs = rmse_random_lcs.mean()
        self.rmse_random_unmonitored = rmse_random_unmonitored.mean()
        
        
    def Dopt_placement_estimation(self):
        """
        Compute estimated signal (and errors) from distribution obtained with D-optimal criterion at given variances ratio.
        The estimation uses GLS and errors are RMSE,MAE.

        Returns
        -------
        None.

        """
        
        sensor_placement = SP.SensorPlacement('D_optimal', self.n, self.r, self.p_zero_estimate, self.p_eps_estimate,
                                              self.p_empty, self.var_eps, self.var_zero)
        
        sensor_placement.initialize_problem(self.Psi,self.alpha_reg)
        try:
            sensor_placement.LoadLocations(self.Dopt_path, self.alpha_reg, self.var_zero)
            
            sensor_placement.locations = sensor_placement.dict_locations[self.p_zero_estimate]
            sensor_placement.weights = sensor_placement.dict_weights[self.p_zero_estimate]
        except:
            sensor_placement.locations = [np.zeros(self.n),np.zeros(self.n),np.zeros(self.n)]
            sensor_placement.weights = [np.zeros(self.n),np.zeros(self.n)]
            
        print(f'Dopt chosen locations for epsilon {self.var_zero} sigma {self.var_eps}\n{sensor_placement.locations}')
        
        if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
            print(f'No solution found for D-optimal with {self.p_zero_estimate} reference stations\n')
            self.rmse_Dopt_full,self.rmse_Dopt_refst,self.rmse_Dopt_lcs,self.rmse_Dopt_unmonitored = np.inf,np.inf,np.inf,np.inf
            
            return
        
        # compute location and covariance matrix
        sensor_placement.C_matrix()
        sensor_placement.covariance_matrix_GLS(self.Psi)
        
        # sensors measurements (with noise)
        y_lcs = self.ds_lcs.loc[:,sensor_placement.locations[0]]
        y_refst = self.ds_refst.loc[:,sensor_placement.locations[1]]
        y_empty = self.ds_real.loc[:,sensor_placement.locations[2]]
        y_real = self.ds_real
        
        # estimated regressor
        sensor_placement.beta_estimated_GLS(self.Psi,y_refst.T,y_lcs.T)
           
        # estimated signal
        y_pred = self.get_estimated_measurements(sensor_placement.beta_hat,y_real)
        
        # compute error metrics
        [self.rmse_Dopt_full,self.rmse_Dopt_refst,self.rmse_Dopt_lcs,self.rmse_Dopt_unmonitored] = self.compute_metrics(
            y_real,
            y_pred,
            y_refst,
            y_lcs, 
            y_empty)
        
        
        
    def Dopt_placement_convergene(self,var_orig=1e0):
        """
        Compute estimated signal (and errors) from distribution obtained with D-optimal criterion for specific variances ratio
        (var_orig).
        Then, re-compute estimations for lower variances ratio but maintaining the sensors distribution.
        The estimations are GLS.
        The errors are RMSE, MAE

        Parameters
        ----------
        var_orig : float , optional
            variances ratio used for distribution. The default is 1e0.

        Returns
        -------
        None.

        """
        
        
        sensor_placement = SP.SensorPlacement('D_optimal', self.n, self.r, self.p_zero_estimate, self.p_eps_estimate,
                                              self.p_empty, self.var_eps, var_orig)
        
        sensor_placement.initialize_problem(self.Psi,self.alpha_reg)
        sensor_placement.LoadLocations(self.Dopt_path, self.alpha_reg, self.var_zero)
        
        sensor_placement.locations = sensor_placement.dict_locations[self.p_zero_estimate]
        sensor_placement.weights = sensor_placement.dict_weights[self.p_zero_estimate]
        print(f'Dopt chosen locations for epsilon {var_orig} sigma {self.var_eps}\n{sensor_placement.locations}')
        
        if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
            print(f'No solution found for D-optimal with {self.p_zero_estimate} reference stations\n')
            self.rmse_Dopt_full,self.rmse_Dopt_refst,self.rmse_Dopt_lcs,self.rmse_Dopt_unmonitored = np.inf,np.inf,np.inf,np.inf
            
            return
        
        # compute location and covariance matrix
        sensor_placement.C_matrix()
        # interrupt! using locations obtained with var_orig change for new var_zero
        sensor_placement.var_zero = self.var_zero
        
        if self.var_zero != 0:
            sensor_placement.covariance_matrix_GLS(self.Psi)
        else:
            sensor_placement.covariance_matrix_limit(self.Psi)
        
        # sensors measurements (with noise)
        y_lcs = self.ds_lcs.loc[:,sensor_placement.locations[0]]
        y_refst = self.ds_refst.loc[:,sensor_placement.locations[1]]
        y_empty = self.ds_real.loc[:,sensor_placement.locations[2]]
        y_real = self.ds_real
        
        # estimated regressor
        if self.var_zero !=0:
            sensor_placement.beta_estimated_GLS(self.Psi,y_refst.T,y_lcs.T)
        else:
            sensor_placement.beta_estimated_limit(self.Psi,y_refst.T,y_lcs.T)
        
        # estimated signal
        y_pred = self.get_estimated_measurements(sensor_placement.beta_hat,y_real)
        
        # compute error metrics
        [self.rmse_Dopt_full,self.rmse_Dopt_refst,self.rmse_Dopt_lcs,self.rmse_Dopt_unmonitored] = self.compute_metrics(
            y_real,
            y_pred,
            y_refst,
            y_lcs, 
            y_empty)
           
        
    def rankMax_placement_estimation(self):
        """
        Compute estimated signal (and errors) using sensors distribution obtained with rankMax criterion.
        The estimations use GLS with specific variances ratio.
        The errors are RMSE,MAE

        Returns
        -------
        None.

        """
        
        
        sensor_placement = SP.SensorPlacement('rankMax', self.n, self.r, self.p_zero_estimate, self.p_eps_estimate,
                                              self.p_empty, self.var_eps, self.var_zero)
        
        sensor_placement.initialize_problem(self.Psi,self.alpha_reg)
        sensor_placement.LoadLocations(self.rank_path, self.alpha_reg, self.var_zero)
        
        sensor_placement.locations = sensor_placement.dict_locations[self.p_zero_estimate]
        sensor_placement.weights = sensor_placement.dict_weights[self.p_zero_estimate]
        print(f'rankMax chosen locations for epsilon {self.var_zero} sigma {self.var_eps}\n{sensor_placement.locations}')
        
        if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
            print(f'No solution found for rankMax with {self.p_zero_estimate} reference stations\n')
            return
        
        # compute location and covariance matrix
        sensor_placement.C_matrix()
        sensor_placement.covariance_matrix_GLS(self.Psi)
        
        # sensors measurements (with noise)
        y_lcs = self.ds_lcs.loc[:,sensor_placement.locations[0]]
        y_refst = self.ds_refst.loc[:,sensor_placement.locations[1]]
        y_empty = self.ds_real.loc[:,sensor_placement.locations[2]]
        y_real = self.ds_real
        
        # estimated regressor
        sensor_placement.beta_estimated_GLS(self.Psi,y_refst.T,y_lcs.T)
        
        # estimated signal
        y_pred = self.get_estimated_measurements(sensor_placement.beta_hat,y_real)
        
        # compute error metrics
        [self.rmse_rankMax_full,self.rmse_rankMax_refst,self.rmse_rankMax_lcs,self.rmse_rankMax_unmonitored] = self.compute_metrics(
            y_real,
            y_pred,
            y_refst,
            y_lcs, 
            y_empty)
        
        
            
        
    def rankMax_placement_estimation_limit(self):
        """
        Compute estimated signal (and errors) using distribution obtained with rankMax criterion.
        The estimations use the limit of the GLS regressor at variances_ratio -> 0

        Returns
        -------
        None.

        """
         
        
        sensor_placement = SP.SensorPlacement('rankMax', self.n, self.r, self.p_zero_estimate, self.p_eps_estimate,
                                              self.p_empty, self.var_eps, self.var_zero)
        
        sensor_placement.initialize_problem(self.Psi,self.alpha_reg)
        sensor_placement.LoadLocations(self.rank_path, self.alpha_reg, self.var_zero)
        
        sensor_placement.locations = sensor_placement.dict_locations[self.p_zero_estimate]
        sensor_placement.weights = sensor_placement.dict_weights[self.p_zero_estimate]
        print(f'rankMax chosen locations for epsilon {self.var_zero} sigma {self.var_eps}\n{sensor_placement.locations}')
        
        if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
            print(f'No solution found for rankMax with {self.p_zero_estimate} reference stations\n')
            return
        
        # compute location and covariance matrix
        sensor_placement.C_matrix()
        sensor_placement.covariance_matrix_limit(self.Psi)
        
        # sensors measurements (with noise)
        y_lcs = self.ds_lcs.loc[:,sensor_placement.locations[0]]
        y_refst = self.ds_refst.loc[:,sensor_placement.locations[1]]
        y_empty = self.ds_real.loc[:,sensor_placement.locations[2]]
        y_real = self.ds_real
        
        # estimated regressor using beta_hat(epsilon->0+)
        sensor_placement.beta_estimated_limit(self.Psi,y_refst.T,y_lcs.T)
        
        # estimated signal
        y_pred = self.get_estimated_measurements(sensor_placement.beta_hat,y_real)
        
        # compute error metrics
        [self.rmse_rankMax_full,self.rmse_rankMax_refst,self.rmse_rankMax_lcs,self.rmse_rankMax_unmonitored] = self.compute_metrics(
            y_real,
            y_pred,
            y_refst,
            y_lcs, 
            y_empty)
        
        
         
  
    def analytical_estimation(self,criterion='rankMax',random_path=''):
        """
        Compute analytical MSE from sensors distributions in the network.

        Parameters
        ----------
        criterion : str
            algorithm used for obtaining locations: ['rankMax','D_optimal']

        Returns
        -------
        None.

        """
     
        
        # load sensor placement results
        
        sensor_placement = SP.SensorPlacement(criterion, self.n, self.r, self.p_zero_estimate, self.p_eps_estimate,
                                              self.p_empty, self.var_eps, self.var_zero)
        
        sensor_placement.initialize_problem(self.Psi,self.alpha_reg)
        if criterion == 'rankMax':
            sensor_placement.LoadLocations(self.rank_path, self.alpha_reg, self.var_zero)
        elif criterion == 'D_optimal':
            sensor_placement.LoadLocations(self.Dopt_path, self.alpha_reg, self.var_zero)
        elif criterion == 'random':
            fname = random_path+f'randomPlacement_locations_r{self.r}_pEmpty{self.p_empty}_numRandomPlacements{self.num_random_placements}.pkl'
            with open(fname,'rb') as f:
                dict_random_locations = pickle.load(f)
            
            # random placement estimation
            random_placement = RP.randomPlacement(self.p_eps_estimate,self.p_zero_estimate,self.p_empty,self.n)
            # obtain locations and covariance matrices
            random_placement.locations = dict_random_locations[self.p_zero_estimate]
            random_placement.C_matrix()
            random_placement.covariance_matrix_GLS(self.Psi, self.var_eps, self.var_zero)
            
         
            
        
        # get solution for specific number of refst in the network
        sensor_placement.locations = sensor_placement.dict_locations[self.p_zero_estimate]
        sensor_placement.weights = sensor_placement.dict_weights[self.p_zero_estimate]
        print(f'rankMax chosen locations for epsilon {self.var_zero} sigma {self.var_eps}\n{sensor_placement.locations}')
        
        # check if solution exists
        if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
            print(f'No solution found for {criterion} with {self.p_zero_estimate} reference stations\n')
            self.mse_analytical_full, self.mse_analytical_refst, self.mse_analytical_lcs, self.mse_analytical_unmonitored = np.inf,np.inf,np.inf,np.inf
            
            return
        
        # get theta matrices
        sensor_placement.C_matrix()
        Theta_lcs = sensor_placement.C[0]@self.Psi
        Theta_refst = sensor_placement.C[1]@self.Psi
        Theta_empty = sensor_placement.C[2]@self.Psi
        
        
        # get estimated regressor covariance matrix
        if self.var_zero!=0.0:
            sensor_placement.covariance_matrix_GLS(self.Psi)
        else:
            sensor_placement.covariance_matrix_limit(self.Psi)
        
        # compute covariance estimated residuals
        cov_full = self.Psi@sensor_placement.Cov@self.Psi.T
        cov_refst = Theta_refst@sensor_placement.Cov@Theta_refst.T
        cov_lcs = Theta_lcs@sensor_placement.Cov@Theta_lcs.T
        cov_empty = Theta_empty@sensor_placement.Cov@Theta_empty.T
        
        
        
        self.mse_analytical_full = np.trace(np.abs(cov_full))/self.n
        self.mse_analytical_refst = np.trace(cov_refst)/self.p_zero_estimate
        self.mse_analytical_lcs = np.trace(cov_lcs)/self.p_eps_estimate
        self.mse_analytical_unmonitored = np.trace(np.abs(cov_empty))/self.p_empty
       
    def analytical_estimation_random(self,random_path):
        """
        Compute analytical MSE from random sensors distributions in the network.

        Parameters
        ----------
        criterion : str
            algorithm used for obtaining locations: ['rankMax','D_optimal']

        Returns
        -------
        None.

        """
     
        
        # load random distributions
        fname = random_path+f'randomPlacement_locations_r{self.r}_pEmpty{self.p_empty}_numRandomPlacements{self.num_random_placements}.pkl'
        with open(fname,'rb') as f:
            dict_random_locations = pickle.load(f)
        
        # random placement estimation
        random_placement = RP.randomPlacement(self.p_eps_estimate,self.p_zero_estimate,self.p_empty,self.n)
        # obtain locations and covariance matrices
        random_placement.locations = dict_random_locations[self.p_zero_estimate]
        random_placement.C_matrix()
        # get regressor covariance matrices
        if self.var_zero !=0:
            random_placement.covariance_matrix_GLS(self.Psi, self.var_eps, self.var_zero)
        else:
            random_placement.covariance_matrix_limit(self.Psi, self.var_eps, self.r)
            
        self.mse_analytical_full_random = {el:[] for el in random_placement.Covariances}
        self.mse_analytical_refst_random = {el:[] for el in random_placement.Covariances}
        self.mse_analytical_lcs_random = {el:[] for el in random_placement.Covariances}
        self.mse_analytical_unmonitored_random = {el:[] for el in random_placement.Covariances}
        
        for idx in random_placement.Covariances:
        
            # get theta matrices
            Theta_lcs = random_placement.C[idx][0]@self.Psi
            Theta_refst = random_placement.C[idx][1]@self.Psi
            Theta_empty = random_placement.C[idx][2]@self.Psi
        
            # compute covariance estimated residuals
            cov_full = self.Psi@random_placement.Covariances[idx]@self.Psi.T
            cov_refst = Theta_refst@random_placement.Covariances[idx]@Theta_refst.T
            cov_lcs = Theta_lcs@random_placement.Covariances[idx]@Theta_lcs.T
            cov_empty = Theta_empty@random_placement.Covariances[idx]@Theta_empty.T
            
        
        
            self.mse_analytical_full_random[idx].append(np.trace(np.abs(cov_full))/self.n)
            self.mse_analytical_refst_random[idx].append(np.trace(cov_refst)/self.p_zero_estimate)
            self.mse_analytical_lcs_random[idx].append(np.trace(cov_lcs)/self.p_eps_estimate)
            self.mse_analytical_unmonitored_random[idx].append(np.trace(np.abs(cov_empty))/self.p_empty)
        
        self.mse_analytical_full_random = self.mean_confidence_interval([i[0] for i in self.mse_analytical_full_random.values()],confidence=0.50) 
        self.mse_analytical_refst_random = self.mean_confidence_interval([i[0] for i in self.mse_analytical_refst_random.values()],confidence=0.50) 
        self.mse_analytical_lcs_random = self.mean_confidence_interval([i[0] for i in self.mse_analytical_lcs_random.values()],confidence=0.50) 
        self.mse_analytical_unmonitored_random = self.mean_confidence_interval([i[0] for i in self.mse_analytical_unmonitored_random.values()],confidence=0.50) 
          
      
    def analytical_estimation_exhaustive(self,exhaustive_locations):
        
        #get locations
        random_placement = RP.randomPlacement(self.p_eps_estimate,self.p_zero_estimate,self.p_empty,self.n)
        random_placement.locations = exhaustive_locations
        
        # get regressor covariance matrices
        random_placement.C_matrix()
        if self.var_zero !=0:
            random_placement.covariance_matrix_GLS(self.Psi, self.var_eps, self.var_zero)
        else:
            random_placement.covariance_matrix_limit(self.Psi, self.var_eps, self.r)
            
        self.mse_analytical_full = {el:[] for el in random_placement.Covariances}
        self.mse_analytical_refst = {el:[] for el in random_placement.Covariances}
        self.mse_analytical_lcs = {el:[] for el in random_placement.Covariances}
        self.mse_analytical_unmonitored = {el:[] for el in random_placement.Covariances}
        
        for idx in random_placement.Covariances:
        
            # get theta matrices
            Theta_lcs = random_placement.C[idx][0]@self.Psi
            Theta_refst = random_placement.C[idx][1]@self.Psi
            Theta_empty = random_placement.C[idx][2]@self.Psi
        
            # compute covariance estimated residuals
            cov_full = self.Psi@random_placement.Covariances[idx]@self.Psi.T
            cov_refst = Theta_refst@random_placement.Covariances[idx]@Theta_refst.T
            cov_lcs = Theta_lcs@random_placement.Covariances[idx]@Theta_lcs.T
            cov_empty = Theta_empty@random_placement.Covariances[idx]@Theta_empty.T
            
        
        
            self.mse_analytical_full[idx].append(np.trace(np.abs(cov_full))/self.n)
            self.mse_analytical_refst[idx].append(np.trace(cov_refst)/self.p_zero_estimate)
            self.mse_analytical_lcs[idx].append(np.trace(cov_lcs)/self.p_eps_estimate)
            self.mse_analytical_unmonitored[idx].append(np.trace(np.abs(cov_empty))/self.p_empty)
        
      
        
        
        
    
    
# =============================================================================
# OLD CODE     
# =============================================================================
        
    
    def compute_estimations(self):
            """
            Estimate signal at ceratin points (unmonitored/refst/LCSs)
    
            Parameters
            ----------
            solutions_path : TYPE
                DESCRIPTION.
    
            Returns
            -------
            None.
    
            """
            sensors_range = np.arange(0,self.n-self.p_empty+1,1)
            # self.dict_estimations_random = {el:np.inf for el in sensors_range}
            # self.dict_estimations = {el:np.inf for el in sensors_range}
            
            self.dict_rmse = {el:np.inf for el in sensors_range}
            self.dict_mae = {el:np.inf for el in sensors_range}
            self.dict_rmse_multi = {el:np.inf for el in sensors_range}
            self.dict_mae_multi = {el:np.inf for el in sensors_range}
            
            self.dict_rmse_random = {el:np.inf for el in sensors_range}
            self.dict_mae_random = {el:np.inf for el in sensors_range}
            
            lowrank_basis = LRB.LowRankBasis(self.ds_estimation, self.r)
            lowrank_basis.snapshots_matrix()
            lowrank_basis.low_rank_decomposition(normalize=True)
            self.Psi = lowrank_basis.Psi.copy()
            
            fname = self.rank_path+f'randomPlacement_locations_r{self.r}_pEmpty{self.p_empty}_numRandomPlacements{self.num_random_placements}.pkl'
            with open(fname,'rb') as f:
                self.dict_random_locations = pickle.load(f)
            
            for p_zero in sensors_range:
                print(f'Estimating for {p_zero} reference stations')
                p_eps = self.n-(p_zero+self.p_empty)
                # random placement
                random_placement = RP.randomPlacement(p_eps,p_zero,self.p_empty,self.n)
                random_placement.locations = self.dict_random_locations[p_zero]
                random_placement.C_matrix()
                random_placement.Cov_metric(self.Psi, self.var_eps, self.var_zero,self.placement_metric)
                random_placement.beta_estimated(self.Psi, self.ds_estimation, self.var_eps, self.var_zero)
                random_placement.estimation(self.Psi,self.ds_estimation,self.var_eps,self.var_zero,self.locations_to_estimate)
                random_placement.estimation_metric(self.ds_estimation)
                random_placement_rmse = np.array([i for i in random_placement.rmse.values()])
                random_placement_mae = np.array([i for i in random_placement.mae.values()])
                self.dict_rmse_random[p_zero] = [random_placement_rmse.mean(),random_placement_rmse.min(),random_placement_rmse.max()]
                self.dict_mae_random[p_zero] = [random_placement_mae.mean(),random_placement_mae.min(),random_placement_mae.max()]
                
                
                # algorithm
                sensor_placement = SP.SensorPlacement(self.solving_algorithm, self.n, self.r, p_zero, p_eps, self.p_empty, self.var_eps, self.var_zero)
                sensor_placement.initialize_problem(self.Psi,self.alpha_reg)
                # use weights and locations from file
                if self.solving_algorithm == 'rankMax':
                    sensor_placement.LoadLocations(self.rank_path, self.alpha_reg, self.var_zero)
                else:
                    sensor_placement.LoadLocations(self.Dopt_path, self.alpha_reg, self.var_zero)
                    
                sensor_placement.locations = sensor_placement.dict_locations[p_zero]
                sensor_placement.weights = sensor_placement.dict_weights[p_zero]
                
                if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
                    print(f'No solution found for {self.solving_algorithm} with {p_zero} reference stations\n')
                    continue
                # compute location and covariance matrix
                sensor_placement.C_matrix()
                sensor_placement.covariance_matrix(self.Psi,metric=self.placement_metric,activate_error_solver=False)
                
                # sensors measurements (with noise)
                y_lcs = self.perturbate_signal(self.ds_estimation.loc[:,sensor_placement.locations[0]],self.var_eps*10,seed=0)
                y_refst = self.perturbate_signal(self.ds_estimation.loc[:,sensor_placement.locations[1]],self.var_zero*10,seed=0)
                y_empty = self.ds_estimation.loc[:,sensor_placement.locations[2]]
                # estimated regressor
                sensor_placement.beta_estimated_GLS(self.Psi,y_refst.T,y_lcs.T)
                # estimate at certain locations
                if self.locations_to_estimate == 'empty':
                    l=2
                    y_real = y_empty
                elif self.locations_to_estimate == 'RefSt':
                    l=1
                    y_real = y_refst
                elif self.locations_to_estimate == 'LCSs':
                    l=0
                    y_real = y_lcs
                elif self.locations_to_estimate == 'All':
                    print('To be included')
                    
                y_hat = sensor_placement.C[l]@self.Psi@sensor_placement.beta_hat
                y_pred = y_hat.T
                y_pred.columns = y_empty.columns
                # self.dict_estimations[p_zero] = y_pred
                
                # metrics
                self.dict_rmse[p_zero] = np.sqrt(mean_squared_error(y_real, y_pred))
                self.dict_mae[p_zero] =  mean_absolute_error(y_real, y_pred)
                
                self.dict_rmse_multi[p_zero] = [np.sqrt(mean_squared_error(y_real.iloc[:,i],y_pred.iloc[:,i])) for i in range(y_real.shape[1])]
                self.dict_mae_multi[p_zero] = [mean_absolute_error(y_real.iloc[:,i],y_pred.iloc[:,i]) for i in range(y_real.shape[1])]
                
                
                
                
            
            
            
            
                
            
        
    def compute_performance_metrics(self,solutions_path):
        """
        Compute performance metrics (rmse,mae) of reconstructing signal at certain locations (unmonitored/refst/lcs)
        
        Load locations obtained with Dopt at certain variances ratio and compute variance matrix/estimator (GLS). Then estimate from measurements
        
        
        Parameters
        ----------
        solutions_path : TYPE
            DESCRIPTION.
    
        Returns
        -------
        None.
    
        """
        
        self.dict_rmse = {el:np.inf for el in self.sensors_range}
        self.dict_mae = {el:np.inf for el in self.sensors_range}
        self.dict_rmse_multi = {el:np.inf for el in self.sensors_range}
        self.dict_mae_multi = {el:np.inf for el in self.sensors_range}
        
        self.dict_rmse_random = {el:np.inf for el in self.sensors_range}
        self.dict_mae_random = {el:np.inf for el in self.sensors_range}
        
        lowrank_basis = LRB.LowRankBasis(self.ds_estimation, self.r)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)
        self.Psi = lowrank_basis.Psi.copy()
        
        for p_zero in self.sensors_range:
            p_eps = self.n-(p_zero+self.p_empty)
            # random placement
            random_placement = RP.randomPlacement(p_eps,p_zero,self.p_empty,self.n)
            random_placement.place_sensors(num_samples=num_random_placements,random_seed=92)
            random_placement.C_matrix()
            random_placement.Cov_metric(self.Psi, self.var_eps, self.var_zero,criteria=self.placement_metric)
            random_placement.beta_estimated(self.Psi, self.ds_estimation, self.var_eps, self.var_zero)
            random_placement.estimation(self.Psi,self.ds_estimation)
            random_placement.estimation_metric(self.ds_estimation)
            random_placement_rmse = np.array([i for i in random_placement.rmse.values()])
            random_placement_mae = np.array([i for i in random_placement.mae.values()])
            self.dict_rmse_random[p_zero] = [random_placement_rmse.mean(),random_placement_rmse.min(),random_placement_rmse.max()]
            self.dict_mae_random[p_zero] = [random_placement_mae.mean(),random_placement_mae.min(),random_placement_mae.max()]
            
            # algorithm placement
            sensor_placement = SP.SensorPlacement(self.solving_algorithm, self.n, self.r, p_zero, p_eps, self.p_empty, self.var_eps, self.var_zero)
            sensor_placement.initialize_problem(self.Psi,self.alpha_reg)
            # use weights and locations from file
            sensor_placement.LoadLocations(solutions_path, self.placement_metric,self.alpha_reg)
            sensor_placement.locations = sensor_placement.dict_locations[p_zero]
            sensor_placement.weights = sensor_placement.dict_weights[p_zero]
            
            if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
                print(f'No solution found for {self.solving_algorithm} with {p_zero} reference stations\n')
                continue
            
            # compute location and covariance matrix
            sensor_placement.C_matrix()
            sensor_placement.covariance_matrix(self.Psi,metric=self.placement_metric,activate_error_solver=False)
            # sensors measurements
            y_lcs = self.ds_estimation.loc[:,sensor_placement.locations[0]]
            y_refst = self.ds_estimation.loc[:,sensor_placement.locations[1]]
            y_empty = self.ds_estimation.loc[:,sensor_placement.locations[2]]
            # estimated regressor
            sensor_placement.beta_estimated(self.Psi,y_refst.T,y_lcs.T)
            # estimate unmonitored locations
            y_hat = sensor_placement.C[2]@self.Psi@sensor_placement.beta_hat
            y_pred = y_hat.T
            y_pred.columns = y_empty.columns
            # metrics
            
            self.dict_rmse[p_zero] = np.sqrt(mean_squared_error(y_empty, y_pred))
            self.dict_mae[p_zero] =  mean_absolute_error(y_empty, y_pred)
            
            self.dict_rmse_multi[p_zero] = [np.sqrt(mean_squared_error(y_empty.iloc[:,i],y_pred.iloc[:,i])) for i in range(y_empty.shape[1])]
            self.dict_mae_multi[p_zero] = [mean_absolute_error(y_empty.iloc[:,i],y_pred.iloc[:,i]) for i in range(y_empty.shape[1])]
            
        
            
    def save_results(self,results_path):
        # random placement
        fname = results_path+f'RMSE_estimation_randomPlacement_vs_p0_r{self.r}_varZero{self.var_zero:.1e}_pEmpty{self.p_empty}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.dict_rmse_random,f)
        
        fname = results_path+f'MAE_estimation_randomPlacement_vs_p0_r{self.r}_varZero{self.var_zero:.1e}_pEmpty{self.p_empty}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.dict_mae_random,f)
        
        # algorithm placement
        fname = results_path+f'RMSE_estimation_{self.solving_algorithm}_vs_p0_r{self.r}_varZero{self.var_zero:.1e}_pEmpty{self.p_empty}'
        if self.solving_algorithm == 'rankMin_reg':
            fname = fname+f'_alpha{self.alpha_reg}.pkl'
        else:
            fname = fname+'.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.dict_rmse,f)
        
        fname = results_path+f'MAE_estimation_{self.solving_algorithm}_vs_p0_r{self.r}_varZero{self.var_zero:.1e}_pEmpty{self.p_empty}'
        if self.solving_algorithm == 'rankMin_reg':
            fname = fname+f'_alpha{self.alpha_reg}.pkl'
        else:
            fname = fname+'.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.dict_mae,f)
            
        fname = results_path+f'RMSE_estimation_multi_{self.solving_algorithm}_vs_p0_r{self.r}_varZero{self.var_zero:.1e}_pEmpty{self.p_empty}'
        if self.solving_algorithm == 'rankMin_reg':
            fname = fname+f'alpha{self.alpha_reg}.pkl'
        else:
            fname = fname+'.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.dict_rmse_multi,f)
        
        fname = results_path+f'MAE_estimation_multi_{self.solving_algorithm}_vs_p0_r{self.r}_varZero{self.var_zero:.1e}_pEmpty{self.p_empty}'
        if self.solving_algorithm == 'rankMin_reg':
            fname = fname+f'_alpha{self.alpha_reg}.pkl'
        else:
            fname = fname+'.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.dict_mae_multi,f)
            
        print(f'Results saved in {results_path}')
            
            
                
            
            
        
        
        
        
if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
      
    print('Loading data set')
    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    dataset_source = 'synthetic' #['synthetic','real']
    
    dataset = DS.DataSet(pollutant, start_date, end_date, files_path,source=dataset_source)
    dataset.load_dataSet()
    dataset.cleanMissingvalues(strategy='stations',tol=0.1)
    dataset.cleanMissingvalues(strategy='remove')
    train_set_end = '2020-12-31 23:00:00'
    val_set_begin = '2021-01-01 00:00:00'
    val_set_end = '2021-12-31 23:00:00'
    test_set_begin = '2022-01-01 00:00:00'
    dataset.train_test_split(train_set_end, val_set_begin, val_set_end,test_set_begin)
     
    # network parameters
    n = dataset.ds.shape[1]
    r = 54 if dataset_source == 'synthetic' else 34
    p_empty = int(n*0.4)
    var_eps,var_zero = 1,1e-6
    alpha_reg = 1e-3
    num_random_placements = 100
    solving_algorithm = 'D_optimal' #['rankMin_reg','D_optimal']
    placement_metric = 'logdet' #['logdet','eigval']
    
    
    
    if dataset_source == 'real':
        solutions_path = results_path+f'Unmonitored_locations/Training_Testing_split/TrainingSet_results/{solving_algorithm}/'
    else:
        solutions_path = results_path+f'Unmonitored_locations/Synthetic_Data/TrainingSet_results/{solving_algorithm}/'
    
    generate_files = False
    
    if generate_files:
        ds_estimation = dataset.ds_test
       
        estimation = Estimation(n, r, p_empty, var_eps, var_zero, solving_algorithm, alpha_reg, dataset_source, 
                                placement_metric, ds_estimation)
        
        estimation.performance_metrics(solutions_path)
        estimation.save_results(results_path)
    
    else:
        #plt.close('all')
        plots = Plots.Plots(save_path=results_path,fs_legend=4,show_plots=True)
        Dopt_path = results_path+f'Unmonitored_locations/Synthetic_Data/TestingSet_results/D_optimal/'
        rank_path = results_path+f'Unmonitored_locations/Synthetic_Data/TestingSet_results/rankMin_reg/'
        plots.plot_other_performance_metrics(Dopt_path,rank_path,r,var_zero,p_empty,alpha_reg,metric='RMSE',save_fig=False)
        #plots.plot_other_performance_metrics(Dopt_path,rank_path,r,var_zero,p_empty,alpha_reg,metric='MAE',save_fig=False)
    
    
    