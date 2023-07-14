#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:54:16 2023

@author: jparedes
"""
import os
import numpy as np
import pandas as pd
import pickle 

import DataSet as DS
import LowRankBasis as LRB
import randomPlacement as RP
import Plots

# =============================================================================
# Sensor placement for heterogeneous monitoring networks
# =============================================================================


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
    p_zero = 10
    p_eps = n-p_zero
    p_empty = 0
    var_eps,var_zero = 1,1*1e-2
    print(f'Network description\n{n} locations in total\n{p_zero} locations for reference stations\n{p_eps} locations for LCSs\n{p_empty} locations unmonitored\n{var_zero} variance of reference stations\n{var_eps} variance of LCSs')
    
    print(f'Low rank basis\nr={r}')
    lowrank_basis = LRB.LowRankBasis(dataset.ds, r)
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    plots = Plots.Plots(save_path=results_path,show_plots=False)
    fig_singularValues, fig_singularValues_cumSum = plots.plot_singular_values(lowrank_basis.S,save_fig=True)

    print('Random sensor placement')
    random_placement = RP.randomPlacement(p_eps,p_zero,p_empty,n)
    random_placement.place_sensors(num_samples=100,random_seed=92)
    random_placement.C_matrix()
    random_placement.design_metric(lowrank_basis.Psi, var_eps, var_zero,criteria='D_optimal')
    
    

