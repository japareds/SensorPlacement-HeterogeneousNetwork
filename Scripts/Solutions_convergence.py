#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 12:54:08 2023

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
# Compute convergence. Use weights/locations from previous iterations
# =============================================================================

def compute_previous_step():
    # obtain data-driven basis
    lowrank_basis = LRB.LowRankBasis(dataset.ds_train, r) # using validation set now
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    
    sensors_range = np.arange(0,n-p_empty+1,1) #np.arange(0,r+1,1)
    dict_results = {el:0 for el in sensors_range}
    dict_results_convex = {el:0 for el in sensors_range}
    
    for p_zero in sensors_range:

        p_eps = n-(p_zero+p_empty)
        print(f'Network description\n{n} locations in total\n{p_zero} locations for reference stations\n{p_eps} locations for LCSs\n{p_empty} locations unmonitored\n{var_zero} variance of reference stations\n{var_eps} variance of LCSs')
        print(f'Low rank basis\nr={r}')
        print('Rank-max proposed sensor placement algorithm')
        
        
        sensor_placement = SP.SensorPlacement(solving_algorithm, n, r, p_zero, p_eps, p_empty, var_eps, var_zero_previous)
        sensor_placement.initialize_problem(lowrank_basis.Psi,alpha=alpha_reg)
        
        # use weights and locations from file
        sensor_placement.LoadLocations(f'{solutions_path}{solving_algorithm}/', placement_metric,alpha_reg)
        sensor_placement.locations = sensor_placement.dict_locations[p_zero]
        sensor_placement.weights = sensor_placement.dict_weights[p_zero]
        
        # compute location and covariance matrices
        ## convex
        precision_matrix = (var_eps**-1)*lowrank_basis.Psi.T@np.diag(sensor_placement.weights[0])@lowrank_basis.Psi + (var_zero**-1)*lowrank_basis.Psi.T@np.diag(sensor_placement.weights[1])@lowrank_basis.Psi
        sensor_placement.Dopt_metric = -1*np.log(np.linalg.det(precision_matrix))
        
        ## discretization
        sensor_placement.C_matrix()
        precision_matrix = (var_eps**-1)*lowrank_basis.Psi.T@sensor_placement.C[0].T@sensor_placement.C[0]@lowrank_basis.Psi + (var_zero**-1)*lowrank_basis.Psi.T@sensor_placement.C[1].T@sensor_placement.C[1]@lowrank_basis.Psi
        results_sensorplacement = -1*np.log(np.linalg.det(precision_matrix))
        
        # save results
        dict_results[p_zero] = results_sensorplacement
        dict_results_convex[p_zero] = sensor_placement.Dopt_metric
        


    print('Saving results')
        
    fname = results_path+f'Discrete_PreviousIteration_{solving_algorithm}_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc{var_zero_previous}_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
    with open(fname,'wb') as f:
        pickle.dump(dict_results,f)
        
    fname = results_path+f'Convex_PreviousIteration_{solving_algorithm}_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc{var_zero_previous}_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
    with open(fname,'wb') as f:
        pickle.dump(dict_results_convex,f)
        
   
    
    

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
    var_eps,var_zero = 1,1e-6 #[1e-4,1e-6]
    var_zero_previous = 1e-4
    n_random=100
    
    alpha_reg = 1e-3
    solving_algorithm = 'D_optimal' #['rankMin_reg','D_optimal']
    placement_metric = 'logdet' #['logdet','eigval']
     
  
    plots = Plots.Plots(save_path=results_path,fs_legend=4,marker_size=1,show_plots=True)
    
    # generate or plot results
    if dataset_source == 'real':
        solutions_path = results_path+f'Unmonitored_locations/Training_Testing_split/TrainingSet_results/'
    else:
        solutions_path = results_path+f'Unmonitored_locations/Synthetic_Data/TrainingSet_results/'
    rank_path = f'{solutions_path}rankMin_reg/'
    Dopt_path = f'{solutions_path}D_optimal/'
    
    
    generate_files = False
    if generate_files:
        compute_previous_step()
    else:
        #plt.close('all')
        plots.plot_previous_solutions_performance(Dopt_path,rank_path,placement_metric,
                                                  r,var_zero,p_empty,n_random,alpha_reg,
                                                  plot_convex=False,save_fig=False)
        
        