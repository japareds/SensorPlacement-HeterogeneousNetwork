#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:54:16 2023

@author: jparedes
"""
import os
import numpy as np
import scipy.stats
import pickle 

import DataSet as DS
import LowRankBasis as LRB
import randomPlacement as RP
import SensorPlacement as SP
import Plots

# =============================================================================
# Sensor placement for heterogeneous monitoring networks
# =============================================================================
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
    print('Loading data set')
    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    
    dataset = DS.DataSet(pollutant, start_date, end_date, files_path)
    dataset.load_dataSet()
    dataset.cleanMissingvalues(strategy='stations',tol=0.1)
    dataset.cleanMissingvalues(strategy='remove')
    
    # network parameters
    n = dataset.ds.shape[1]
    r = 34
    p_empty = 10
    var_eps,var_zero = 1,1*1e-2
    solving_algorithm = 'D_optimal'
    placement_metric = 'D_optimal'
    
    # obtain data-driven basis
    lowrank_basis = LRB.LowRankBasis(dataset.ds, r)
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    plots = Plots.Plots(save_path=results_path,show_plots=False)
    fig_singularValues, fig_singularValues_cumSum = plots.plot_singular_values(lowrank_basis.S,save_fig=True)

    dict_results_random = {el:0 for el in np.arange(0,r+1,1)}
    dict_results = {el:0 for el in np.arange(0,r+1,1)}
    for p_zero in np.arange(0,r+1,1):
        
        p_eps = n-(p_zero+p_empty)
        
        print(f'Network description\n{n} locations in total\n{p_zero} locations for reference stations\n{p_eps} locations for LCSs\n{p_empty} locations unmonitored\n{var_zero} variance of reference stations\n{var_eps} variance of LCSs')
        print(f'Low rank basis\nr={r}')
        
        print('Random sensor placement')
        random_placement = RP.randomPlacement(p_eps,p_zero,p_empty,n)
        random_placement.place_sensors(num_samples=100,random_seed=92)
        random_placement.C_matrix()
        random_placement.Cov_metric(lowrank_basis.Psi, var_eps, var_zero,criteria=placement_metric)
        
        print('Sensor placement algorithm')
        if p_empty != 0:
            sensor_placement = SP.SensorPlacement(solving_algorithm, n, r, p_zero, p_eps, p_empty, var_eps, var_zero)
        else:
            sensor_placement = SP.SensorPlacement('D_optimal', n, r, p_zero, p_eps, p_empty, var_eps, var_zero)
        sensor_placement.initialize_problem(lowrank_basis.Psi,alpha=1e-1)
        sensor_placement.solve()
        sensor_placement.convertSolution()
        # sensor_placement.locations[0] = [i for i in range(n) if i not in sensor_placement.locations[1]]
        # sensor_placement.p_eps = n - sensor_placement.p_zero
        sensor_placement.C_matrix()
        sensor_placement.covariance_matrix(lowrank_basis.Psi,metric=placement_metric)
        
        results_random = np.array([i for i in random_placement.metric.values()])
        results_sensorplacement = sensor_placement.metric
        
        
        dict_results_random[p_zero] = [results_random.mean(),results_random.min(),results_random.max()]
        dict_results[p_zero] = results_sensorplacement
    
    fname = results_path+f'{placement_metric}_metric_vs_p0_r{r}_sigmaZero{var_zero}_unmonitored2class_pEmpty{p_empty}.pkl'
    with open(fname,'wb') as f:
        pickle.dump(dict_results,f)
        
    fname = results_path+f'randomPlacement_{placement_metric}_metric_vs_p0_r{r}_sigmaZero{var_zero}_unmonitored_pEmpty{p_empty}.pkl'
    with open(fname,'wb') as f:
        pickle.dump(dict_results_random,f)
    
    

