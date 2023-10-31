#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 10:14:31 2023

@author: jparedes
"""


import os
import numpy as np
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
    def __init__(self,n,r,p_empty,p_zero_estimate,var_eps,var_zero,num_random_placements,alpha_reg,solving_algorithm,placement_metric,ds_estimation,Psi,Dopt_path,rank_path,locations_to_estimate):
        self.n = n
        self.r = r
        self.p_empty =p_empty
        self.p_zero_estimate = p_zero_estimate
        self.var_eps = var_eps
        self.var_zero = var_zero
        self.solving_algorithm = solving_algorithm
        self.alpha_reg = alpha_reg
        self.placement_metric = placement_metric
        self.ds_estimation = ds_estimation
        self.Psi = Psi
        self.num_random_placements = num_random_placements
        self.Dopt_path = Dopt_path
        self.rank_path = rank_path
        self.locations_to_estimate = locations_to_estimate
        
        
    
    def perturbate_signal(self,ds_signal,variance,seed):
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0,scale=variance,size=ds_signal.shape)
        return ds_signal+noise
        
    def random_placement_estimation(self, random_path):
        
       
        
        fname = random_path+f'randomPlacement_locations_r{self.r}_pEmpty{self.p_empty}_numRandomPlacements{self.num_random_placements}.pkl'
        with open(fname,'rb') as f:
            dict_random_locations = pickle.load(f)
            
        p_eps_estimate = self.n-(self.p_zero_estimate+self.p_empty)
        
        # random placement estimation
        random_placement = RP.randomPlacement(p_eps_estimate,self.p_zero_estimate,self.p_empty,self.n)
        random_placement.locations = dict_random_locations[self.p_zero_estimate]
        random_placement.C_matrix()
        random_placement.Cov_metric(self.Psi, self.var_eps, self.var_zero,self.placement_metric)
        random_placement.beta_estimated(self.Psi, self.ds_estimation, self.var_eps, self.var_zero)
        random_placement.estimation(self.Psi,self.ds_estimation,self.var_eps,self.var_zero,self.locations_to_estimate)
        rmse_random_placement = np.array([i for i in random_placement.rmse.values()])
        mae_random_placement = np.array([i for i in random_placement.mae.values()])
        self.rmse_best_random = rmse_random_placement.min()
        self.mae_best_random = mae_random_placement.min()
        
    def Dopt_placement_estimation(self):
        p_eps_estimate = self.n-(self.p_zero_estimate+self.p_empty)
        sensor_placement = SP.SensorPlacement('D_optimal', self.n, self.r, self.p_zero_estimate, p_eps_estimate,
                                              self.p_empty, self.var_eps, self.var_zero)
        
        sensor_placement.initialize_problem(self.Psi,self.alpha_reg)
        sensor_placement.LoadLocations(self.Dopt_path, self.alpha_reg, self.var_zero)
        
        sensor_placement.locations = sensor_placement.dict_locations[self.p_zero_estimate]
        sensor_placement.weights = sensor_placement.dict_weights[self.p_zero_estimate]
        print(f'Dopt chosen locations for epsilon {self.var_zero} sigma {self.var_eps}\n{sensor_placement.locations}')
        
        if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
            print(f'No solution found for {self.solving_algorithm} with {self.p_zero_estimate} reference stations\n')
            self.rmse_Dopt = np.inf
            self.mae_Dopt =  np.inf
            
            self.rmse_stations_Dopt = [np.inf]
            self.mae_stations_Dopt = [np.inf]
            
            return
        
        # compute location and covariance matrix
        sensor_placement.C_matrix()
        sensor_placement.covariance_matrix(self.Psi,metric=self.placement_metric,activate_error_solver=False)
        
        # sensors measurements (with noise)
        y_lcs = self.perturbate_signal(self.ds_estimation.loc[:,sensor_placement.locations[0]],self.var_eps*15,seed=0)
        y_refst = self.perturbate_signal(self.ds_estimation.loc[:,sensor_placement.locations[1]],self.var_zero*15,seed=0)
        y_empty = self.ds_estimation.loc[:,sensor_placement.locations[2]]
        
        # estimated regressor
        sensor_placement.beta_estimated_GLS(self.Psi,y_refst.T,y_lcs.T)
        # estimate at certain locations
        if self.locations_to_estimate == 'empty':
            l=2
            y_real = y_empty
            C = sensor_placement.C[l]
        elif self.locations_to_estimate == 'RefSt':
            l=1
            y_real = y_refst
            C = sensor_placement.C[l]
        elif self.locations_to_estimate == 'LCSs':
            l=0
            y_real = y_lcs
            C = sensor_placement.C[l]
        elif self.locations_to_estimate == 'All':
            y_real = self.ds_estimation
            C = np.identity(self.n)
            
        y_hat = C@self.Psi@sensor_placement.beta_hat
        y_pred = y_hat.T
        y_pred.columns = y_real.columns
        
        # compute metric
        # metrics
        self.rmse_Dopt = np.sqrt(mean_squared_error(y_real, y_pred))
        self.mae_Dopt =  mean_absolute_error(y_real, y_pred)
        
        self.rmse_stations_Dopt = [np.sqrt(mean_squared_error(y_real.iloc[:,i],y_pred.iloc[:,i])) for i in range(y_real.shape[1])]
        self.mae_stations_Dopt = [mean_absolute_error(y_real.iloc[:,i],y_pred.iloc[:,i]) for i in range(y_real.shape[1])]
        
    def rankMax_placement_estimation(self):
        
        p_eps_estimate = self.n-(self.p_zero_estimate+self.p_empty)
        sensor_placement = SP.SensorPlacement('rankMax', self.n, self.r, self.p_zero_estimate, p_eps_estimate,
                                              self.p_empty, self.var_eps, self.var_zero)
        
        sensor_placement.initialize_problem(self.Psi,self.alpha_reg)
        sensor_placement.LoadLocations(self.rank_path, self.alpha_reg, self.var_zero)
        
        sensor_placement.locations = sensor_placement.dict_locations[self.p_zero_estimate]
        sensor_placement.weights = sensor_placement.dict_weights[self.p_zero_estimate]
        print(f'rankMax chosen locations for epsilon {self.var_zero} sigma {self.var_eps}\n{sensor_placement.locations}')
        
        if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
            print(f'No solution found for {self.solving_algorithm} with {self.p_zero_estimate} reference stations\n')
            return
        
        # compute location and covariance matrix
        sensor_placement.C_matrix()
        sensor_placement.covariance_matrix(self.Psi,metric=self.placement_metric,activate_error_solver=False)
        
        # sensors measurements (with noise)
        y_lcs = self.perturbate_signal(self.ds_estimation.loc[:,sensor_placement.locations[0]],self.var_eps*15,seed=0)
        y_refst = self.perturbate_signal(self.ds_estimation.loc[:,sensor_placement.locations[1]],self.var_zero*15,seed=0)
        y_empty = self.ds_estimation.loc[:,sensor_placement.locations[2]]
        
        # estimated regressor
        sensor_placement.beta_estimated_GLS(self.Psi,y_refst.T,y_lcs.T)
        # estimate at certain locations
        if self.locations_to_estimate == 'empty':
            l=2
            y_real = y_empty
            C = sensor_placement.C[l]
        elif self.locations_to_estimate == 'RefSt':
            l=1
            y_real = y_refst
            C = sensor_placement.C[l]
        elif self.locations_to_estimate == 'LCSs':
            l=0
            y_real = y_lcs
            C = sensor_placement.C[l]
        elif self.locations_to_estimate == 'All':
            y_real = self.ds_estimation
            C = np.identity(self.n)
            
        y_hat = C@self.Psi@sensor_placement.beta_hat
        y_pred = y_hat.T
        y_pred.columns = y_real.columns
        
        # compute metric
        # metrics
        self.rmse_rankMax = np.sqrt(mean_squared_error(y_real, y_pred))
        self.mae_rankMax =  mean_absolute_error(y_real, y_pred)
        
        self.rmse_stations_rankMax = [np.sqrt(mean_squared_error(y_real.iloc[:,i],y_pred.iloc[:,i])) for i in range(y_real.shape[1])]
        self.mae_stations_rankMax = [mean_absolute_error(y_real.iloc[:,i],y_pred.iloc[:,i]) for i in range(y_real.shape[1])]
        
    def rankMax_placement_estimation_limit(self):
        """
        dataset estimation using measurements at locations determined with rankMax criterion
        The estimation uses the estimated_regressor (beta_hat) for the limit variances_ratio -> 0

        Returns
        -------
        None.

        """
         
        p_eps_estimate = self.n-(self.p_zero_estimate+self.p_empty)
        sensor_placement = SP.SensorPlacement('rankMax', self.n, self.r, self.p_zero_estimate, p_eps_estimate,
                                              self.p_empty, self.var_eps, self.var_zero)
        
        sensor_placement.initialize_problem(self.Psi,self.alpha_reg)
        sensor_placement.LoadLocations(self.rank_path, self.alpha_reg, self.var_zero)
        
        sensor_placement.locations = sensor_placement.dict_locations[self.p_zero_estimate]
        sensor_placement.weights = sensor_placement.dict_weights[self.p_zero_estimate]
        print(f'rankMax chosen locations for epsilon {self.var_zero} sigma {self.var_eps}\n{sensor_placement.locations}')
        
        if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
            print(f'No solution found for {self.solving_algorithm} with {self.p_zero_estimate} reference stations\n')
            return
        
        # compute location and covariance matrix
        sensor_placement.C_matrix()
        sensor_placement.covariance_matrix_limit(self.Psi)
        
        # sensors measurements (with noise)
        y_lcs = self.perturbate_signal(self.ds_estimation.loc[:,sensor_placement.locations[0]],self.var_eps*15,seed=0)
        y_refst = self.perturbate_signal(self.ds_estimation.loc[:,sensor_placement.locations[1]],self.var_zero*15,seed=0)
        y_empty = self.ds_estimation.loc[:,sensor_placement.locations[2]]
        
        # estimated regressor using beta_hat(epsilon->0+)
        sensor_placement.beta_estimated_limit(self.Psi,y_refst.T,y_lcs.T)
        # estimate at certain locations
        if self.locations_to_estimate == 'empty':
            l=2
            y_real = y_empty
            C = sensor_placement.C[l]
        elif self.locations_to_estimate == 'RefSt':
            l=1
            y_real = y_refst
            C = sensor_placement.C[l]
        elif self.locations_to_estimate == 'LCSs':
            l=0
            y_real = y_lcs
            C = sensor_placement.C[l]
        elif self.locations_to_estimate == 'All':
            y_real = self.ds_estimation
            C = np.identity(self.n)
            
        y_hat = C@self.Psi@sensor_placement.beta_hat
        y_pred = y_hat.T
        y_pred.columns = y_real.columns
        
        # compute metric
        # metrics
        self.rmse_rankMax = np.sqrt(mean_squared_error(y_real, y_pred))
        self.mae_rankMax =  mean_absolute_error(y_real, y_pred)
        
        self.rmse_stations_rankMax = [np.sqrt(mean_squared_error(y_real.iloc[:,i],y_pred.iloc[:,i])) for i in range(y_real.shape[1])]
        self.mae_stations_rankMax = [mean_absolute_error(y_real.iloc[:,i],y_pred.iloc[:,i]) for i in range(y_real.shape[1])]
         
    
    
    
    
        
    
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
    
    
    