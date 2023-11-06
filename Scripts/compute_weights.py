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

class Placement():
    def __init__(self,n,p_empty,r,ds_basis,solving_algorithm,var_eps,var_zero,num_random_placements,alpha_reg):
        self.n = n
        self.p_empty = p_empty
        self.r = r
        self.var_eps = var_eps
        self.var_zero = var_zero
        self.solving_algorithm = solving_algorithm
        self.ds_basis = ds_basis
        self.num_random_placements = num_random_placements
        self.alpha_reg = alpha_reg
        
    def placement_random(self):
        """
        Random sensor placement.
        Iterate over different proportions of reference stations and LCSs
        and obtain different combinations of monitored locations.

        Returns
        -------
        None.

        """
        # obtain data-driven basis
        lowrank_basis = LRB.LowRankBasis(self.ds_basis, self.r)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)
        range_sensors = np.arange(0,self.n-self.p_empty+1,1)
        self.dict_locations_random = {el:0 for el in range_sensors}
        
        for p_zero in range_sensors:
            p_eps = self.n-(p_zero + self.p_empty)
            print('Random sensor placement')
            random_placement = RP.randomPlacement(p_eps,p_zero,self.p_empty,self.n)
            random_placement.place_sensors(self.num_random_placements,random_seed=92)
            # random_placement.C_matrix()
            # # random_placement.Cov_metric(lowrank_basis.Psi, var_eps, var_zero,criteria=placement_metric)
            # # results_random = np.array([i for i in random_placement.metric.values()])
            self.dict_locations_random[p_zero] = random_placement.locations
            
      
        
    
        
    def placement_allStations(self):
        """
        Sensor placement for varying number of reference stations and LCSs
        The number of unmonitored locations is fixed a priori (p_empty).
        

        Returns
        -------
        None.

        """
        # obtain data-driven basis
        lowrank_basis = LRB.LowRankBasis(self.ds_basis, self.r)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)
    
        # results
        range_sensors = np.arange(0,self.n-self.p_empty+1,1)#np.arange(0,r+1,1)
                
        
        self.dict_weights = {el:0 for el in range_sensors}
        self.dict_locations = {el:0 for el in range_sensors}
        self.dict_execution_time = {el:0 for el in range_sensors}
        
        for p_zero in range_sensors: # solve for different number of ref.st. (and LCSs)
            
            p_eps = self.n-(p_zero + self.p_empty)
            print(f'Network description\n{self.n} locations in total\n{p_zero} locations for reference stations\n{p_eps} locations for LCSs\n{self.p_empty} locations unmonitored\n{self.var_zero} variance of reference stations\n{self.var_eps} variance of LCSs')
            print(f'Low rank basis\nr={self.r}')
            
            
            print(f'Sensor placement algorithm {self.solving_algorithm}')
            sensor_placement = SP.SensorPlacement(self.solving_algorithm, 
                                                  self.n, self.r,
                                                  p_zero, p_eps, self.p_empty, 
                                                  self.var_eps,self.var_zero)
            
            sensor_placement.initialize_problem(lowrank_basis.Psi,self.alpha_reg)
            sensor_placement.solve()
            sensor_placement.discretize_solution()
            # sensor_placement.C_matrix()
            # sensor_placement.covariance_matrix(lowrank_basis.Psi,metric=placement_metric)
            
            # save results
            # dict_results_random[p_zero] = [results_random.mean(),results_random.min(),results_random.max()]
            
            # if solving_algorithm == 'D_optimal':
            #     if type(sensor_placement.problem.value) == type(None):# error when solving
            #         dict_results_convex_problem[p_zero] = np.inf
            #     else:
            #         dict_results_convex_problem[p_zero] = sensor_placement.problem.value
            # elif solving_algorithm == 'rankMax':
            #     sensor_placement.compute_Doptimal(lowrank_basis.Psi,alpha_reg)
            #     dict_results_convex_problem[p_zero] = [sensor_placement.problem.value,sensor_placement.Dopt_metric]
            #     dict_results_objectives[p_zero] = [sensor_placement.logdet_eps,sensor_placement.trace_zero]
                
            # dict_results_cov[p_zero] = sensor_placement.metric
            self.dict_weights[p_zero] = [sensor_placement.h_eps.value,sensor_placement.h_zero.value]
            self.dict_locations[p_zero] = sensor_placement.locations
            self.dict_execution_time[p_zero] = sensor_placement.exec_time
         

    def save_results(self,results_path):
        
        # random locations
        fname = results_path+f'randomPlacement_locations_r{self.r}_pEmpty{self.p_empty}_numRandomPlacements{self.num_random_placements}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.dict_locations_random,f)
            
            
        fname = results_path+f'Weights_{self.solving_algorithm}_vs_p0_r{self.r}_pEmpty{self.p_empty}_'
        if self.solving_algorithm == 'rankMax':
            fname+=f'alpha{self.alpha_reg:.1e}.pkl'
        else:
            fname+=f'varZero{self.var_zero:.1e}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.dict_weights,f)
    
        fname = results_path+f'DiscreteLocations_{self.solving_algorithm}_vs_p0_r{self.r}_pEmpty{self.p_empty}_'
        if self.solving_algorithm == 'rankMax':
            fname+=f'alpha{self.alpha_reg:.1e}.pkl'
        else:
            fname+=f'varZero{self.var_zero:.1e}.pkl'
        
        with open(fname,'wb') as f:
            pickle.dump(self.dict_locations,f)
    
        fname = results_path+f'ExecutionTime_{self.solving_algorithm}_vs_p0_r{self.r}_pEmpty{self.p_empty}_'
        if self.solving_algorithm == 'rankMax':
            fname+=f'alpha{self.alpha_reg:.1e}.pkl'
        else:
            fname+=f'varZero{self.var_zero:.1e}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.dict_execution_time,f)
        
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
    var_eps,var_zero = 1,1e0
    alpha_reg = 1e-3
    num_random_placements = 100
    solving_algorithm = 'D_optimal' #['rankMax','D_optimal']
    
    compute_weights = False
    if compute_weights:
        placement = Placement(n, p_empty, r, dataset.ds_train, solving_algorithm, var_eps, var_zero, num_random_placements, alpha_reg)
        placement.placement_allStations()
        placement.save_results(results_path)
    else:
        Dopt_path = results_path+f'Synthetic_Data/TrainingSet_results/Doptimal/'
        rank_path = results_path+f'Synthetic_Data/TrainingSet_results/rankMax/'
        p_zero_plot = 30
        plots = Plots.Plots(save_path=results_path,marker_size=1,fs_label=3,fs_legend=4,fs_title=10,show_plots=True)
        plots.plot_weights_evolution(Dopt_path,rank_path,r,n,p_empty,solving_algorithm,p_zero_plot,alpha_reg)
        plots.plot_locations_evolution(Dopt_path,rank_path,r,n,p_empty,solving_algorithm,p_zero_plot,alpha_reg)
   
