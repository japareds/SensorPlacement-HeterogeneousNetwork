#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 12:40:57 2023

@author: jparedes
"""
import os
import numpy as np
import pickle

import DataSet as DS
import LowRankBasis as LRB
import randomPlacement as RP
import SensorPlacement as SP
import Plots
from matplotlib import pyplot as plt

# =============================================================================
# Compute sensor placement metrics on hold-out sets (validation/test)
# =============================================================================


class Validate():
    def __init__(self,n,p_empty,r,alpha_reg,ds_val,train_path,num_random_placements,var_eps,var_zero,placement_metric,solving_algorithm):
        self.n = n
        self.p_empty = p_empty
        self.r = r
        self.alpha_reg = alpha_reg
        self.ds_val = ds_val
        self.train_path = train_path
        self.num_random_placements = num_random_placements
        self.var_eps = var_eps
        self.var_zero = var_zero
        self.placement_metric = placement_metric
        self.solving_algorithm = solving_algorithm
        
        
            
    def compute_validation(self):
        """
        Load weights from rankMax algorithm obtained from training dataset and 
        compute covariance matrix (with both weights and discrete) for different values of 
        regularization parameter alpha_reg

        Returns
        -------
        None.

        """
      
        # obtain data-driven basis from fold dataset
        lowrank_basis = LRB.LowRankBasis(self.ds_val, self.r)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)
       
        
        sensors_range = np.arange(0,self.n-self.p_empty+1,1)
        self.dict_metric_random = {el:0 for el in sensors_range}
        self.dict_results_convex = {el:0 for el in sensors_range}
        self.dict_results_discrete = {el:0 for el in sensors_range}
        self.dict_results_objectives = {el:0 for el in sensors_range}
        
        fname = self.train_path+f'randomPlacement_locations_r{self.r}_pEmpty{self.p_empty}_numRandomPlacements{self.num_random_placements}.pkl'
        with open(fname,'rb') as f:
            dict_random_locations = pickle.load(f)
        
        
        for p_zero in sensors_range:
    
            p_eps = self.n-(p_zero+self.p_empty)
            
            print('Random sensor placement')
                
            random_placement = RP.randomPlacement(p_eps,p_zero,self.p_empty,self.n)
            random_placement.locations = dict_random_locations[p_zero]
            random_placement.C_matrix()
            random_placement.Cov_metric(lowrank_basis.Psi, self.var_eps, self.var_zero,self.placement_metric)
            results_random = np.array([i for i in random_placement.metric.values()])
            
        
            print('Rank-max sensor placement algorithm')
            sensor_placement = SP.SensorPlacement(self.solving_algorithm, self.n, self.r, p_zero, p_eps, self.p_empty, self.var_eps, self.var_zero)
            sensor_placement.initialize_problem(lowrank_basis.Psi,self.alpha_reg)
            
            # use weights and locations from file
            sensor_placement.LoadLocations(self.train_path,self.alpha_reg)
            sensor_placement.locations = sensor_placement.dict_locations[p_zero]
            sensor_placement.weights = sensor_placement.dict_weights[p_zero]
            
            # compute location and covariance matrices
            ## convex
            precision_matrix = (sensor_placement.var_eps**-1)*lowrank_basis.Psi.T@np.diag(sensor_placement.weights[0])@lowrank_basis.Psi + (sensor_placement.var_zero**-1)*lowrank_basis.Psi.T@np.diag(sensor_placement.weights[1])@lowrank_basis.Psi
            sensor_placement.convex_metric = -1*np.log(np.linalg.det(precision_matrix))
            ## objective functions rankMax
            sensor_placement.logdet_eps = np.log(np.linalg.det(lowrank_basis.Psi.T@np.diag(sensor_placement.weights[0])@lowrank_basis.Psi))
            sensor_placement.trace_zero = self.alpha_reg*np.trace(lowrank_basis.Psi.T@np.diag(sensor_placement.weights[0])@lowrank_basis.Psi)
            
            
            ## discretization
            
            if sensor_placement.convex_metric == np.inf or sensor_placement.convex_metric == -np.inf:
                sensor_placement.discrete_metric = np.inf
            else:
                sensor_placement.C_matrix()
                sensor_placement.covariance_matrix(lowrank_basis.Psi,metric=self.placement_metric,activate_error_solver=False)
                sensor_placement.discrete_metric = sensor_placement.metric
            
            # save results
            
            self.dict_metric_random[p_zero] = [results_random.mean(),results_random.min(),results_random.max()]
            self.dict_results_discrete[p_zero] = sensor_placement.discrete_metric
            self.dict_results_convex[p_zero] = sensor_placement.convex_metric
            self.dict_results_objectives[p_zero] = [sensor_placement.logdet_eps,sensor_placement.trace_zero]
            
    def save_results(self,results_path):
        """
        Save metric (convex and discrete logdet) to file

        Parameters
        ----------
        results_path : str
            path to files

        Returns
        -------
        None.

        """
    
   
        fname = results_path+f'randomPlacement_{self.placement_metric}_vs_p0_r{self.r}_varZero{self.var_zero}_pEmpty{self.p_empty}_randomPlacements{self.num_random_placements}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.dict_metric_random,f)
            
        fname = results_path+f'Convex_{self.placement_metric}_vs_p0_r{self.r}_varZero{self.var_zero}_pEmpty{self.p_empty}_alpha{self.alpha_reg}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.dict_results_convex,f)
        
        fname = results_path+f'Discrete_{self.placement_metric}_vs_p0_r{self.r}_varZero{self.var_zero}_pEmpty{self.p_empty}_alpha{self.alpha_reg}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.dict_results_discrete,f)
        
        fname = results_path+f'Objective_{self.placement_metric}_vs_p0_r{self.r}_varZero{self.var_zero}_pEmpty{self.p_empty}_alpha{self.alpha_reg}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.dict_results_objectives,f)
        
        print(f'Results saved in {results_path}')
       
            
    def plot_validation():
        plots = Plots.Plots(save_path=results_path,fs_legend=4,show_plots=True)
        
        train_files_path = results_path +'Unmonitored_locations/Synthetic_Data/TrainingSet_results/rankMin_reg/' if dataset_source == 'synthetic' else results_path+'Unmonitored_locations/Training_Testing_split/TrainingSet_results/'
        val_files_path =  results_path+'Unmonitored_locations/Synthetic_Data/ValidationSet_results/' if dataset_source=='synthetic' else results_path+'Unmonitored_locations/Training_Testing_split/ValidationSet_results/'
        plots.plot_regularization_vs_refst(val_files_path,solving_algorithm,placement_metric,r,var_zero,p_empty,save_fig=False)
        plots.plot_regularization_train_val(train_files_path,val_files_path,solving_algorithm,placement_metric,r,p_empty,var_zero,alpha_reg,save_fig=False)
        p_zero = 30
        plots.plot_regularization_vs_variance(train_files_path,val_files_path,solving_algorithm,placement_metric,r,p_empty,p_zero,alpha_range=[1e-3,1e-2,1e-1],save_fig=False)
        plots.plot_regularization_vs_variance(train_files_path,val_files_path,solving_algorithm,placement_metric,r,p_empty,p_zero,alpha_range=[1e-1,1e0,1e1],save_fig=False)
        plots.plot_regularization_vs_variance(train_files_path,val_files_path,solving_algorithm,placement_metric,r,p_empty,p_zero,alpha_range=[1e1,1e2,1e3],save_fig=False)
    
        plots.plot_validation_convex_metrics(val_files_path,solving_algorithm,placement_metric,r,p_empty,var_zero,alpha_reg,save_fig=False)
        return
    
    def plot_test():
        plots = Plots.Plots(save_path=results_path,fs_legend=4,show_plots=True)
        Dopt_path = f'{results_path}Unmonitored_locations/Synthetic_Data/TestingSet_results/D_optimal/'
        rank_path = f'{results_path}Unmonitored_locations/Synthetic_Data/TestingSet_results/rankMin_reg/'
        plots.plot_convexOpt_results(Dopt_path,rank_path, placement_metric, r, var_zero, p_empty, num_random_placements, alpha_reg,fold='Test',save_fig=False)
        return
    
        
    
    

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
    solving_algorithm = 'rankMax' #['rankMax','D_optimal']
    placement_metric = 'logdet' #['logdet','eigval']
     
    
    # generate validation/test set metric or plot results
    generate_files = True
    if generate_files:
        if dataset_source == 'real':
            train_path = results_path+f'Unmonitored_locations/Training_Testing_split/TrainingSet_results/{solving_algorithm}/'
        else:
            train_path = results_path+f'Synthetic_Data/TrainingSet_results/{solving_algorithm}/'
        
        validation = Validate(n,p_empty,r,alpha_reg,
                              dataset.ds_val, train_path,
                              num_random_placements,
                              var_eps, var_zero,
                              placement_metric,solving_algorithm)
        
        validation.compute_validation()
        validation.save_results(results_path)
        
   
        
    
         