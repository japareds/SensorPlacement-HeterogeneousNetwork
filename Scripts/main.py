#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:42:32 2023

@author: jparedes
"""


import os
import numpy as np
import pickle 
import pandas as pd

import DataSet as DS
import LowRankBasis as LRB
import randomPlacement as RP
import SensorPlacement as SP
import Plots

import compute_weights as CW
import validationTest as VT
import Estimation
import matplotlib.pyplot as plt

class PlacementSolutions():
    def __init__(self,Dopt_path,rank_path):
        self.Dopt_path = Dopt_path
        self.rank_path = rank_path
        
    def compute_logdet_configuration(self,num_random_placements,ds,r,n,p_zero,p_empty,save_fig):
        lowrank_basis = LRB.LowRankBasis(ds, r)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)
        
        variances = [1e-6,1e-4,1e-2,1e-1,1e0]
        dict_logdet_Dopt_var = {el:np.inf for el in variances}
        dict_logdet_rankMax_var = {el:np.inf for el in variances}
        dict_logdet_random_var = {el:np.inf for el in variances}
        fname = Dopt_path+f'randomPlacement_locations_r{r}_pEmpty{p_empty}_numRandomPlacements{num_random_placements}.pkl'
        with open(fname,'rb') as f:
            dict_random_locations = pickle.load(f)
            
        for var in variances:
            # random
            random_placement = RP.randomPlacement(n-(p_zero+p_empty),p_zero,p_empty,n)
            random_placement.locations = dict_random_locations[p_zero]
            random_placement.C_matrix()
            random_placement.Cov_metric(lowrank_basis.Psi, var_eps, var,placement_metric)
            dict_logdet_random_var[var] = np.min([i for i in random_placement.metric.values()])
            
            # Doptimal
            sensor_placement = SP.SensorPlacement('D_optimal', n, r, p_zero,n-(p_zero+p_empty),
                                                  p_empty, var_eps, var)
            
            sensor_placement.initialize_problem(lowrank_basis.Psi,alpha_reg)
            sensor_placement.LoadLocations(Dopt_path, alpha_reg, var)
            
            sensor_placement.locations = sensor_placement.dict_locations[p_zero]
            sensor_placement.weights = sensor_placement.dict_weights[p_zero]
          
            if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
                print(f'No solution found for Dopt with {p_zero} reference stations\n')
                
            
            sensor_placement.C_matrix()
            sensor_placement.covariance_matrix(lowrank_basis.Psi,metric=placement_metric,activate_error_solver=False)
            dict_logdet_Dopt_var[var] = sensor_placement.metric
            
            # rankMax
            sensor_placement = SP.SensorPlacement('rankMax', n, r, p_zero,n-(p_zero+p_empty),
                                                  p_empty, var_eps, var)
            
            sensor_placement.initialize_problem(lowrank_basis.Psi,alpha_reg)
            sensor_placement.LoadLocations(rank_path, alpha_reg, var)
            
            sensor_placement.locations = sensor_placement.dict_locations[p_zero]
            sensor_placement.weights = sensor_placement.dict_weights[p_zero]
          
            if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
                print(f'No solution found for Dopt with {p_zero} reference stations\n')
                
            
          
            sensor_placement.C_matrix()
            sensor_placement.covariance_matrix(lowrank_basis.Psi,metric=placement_metric,activate_error_solver=False)
            dict_logdet_rankMax_var[var] = sensor_placement.metric
            
        
        plots.plot_logdet_vs_variances(dict_logdet_random_var, dict_logdet_Dopt_var, dict_logdet_rankMax_var,
                                       r, n, p_zero, p_empty,save_fig)    
        
    def compare_similarities_locations(self,Dopt_path,ds,r,n,p_empty,p_zero,var_eps,alpha_reg,locations_to_compare='empty'):
        """
        Load locations obtained with Doptimal for different variances and compare the percentage of similarity
        with results with Doptimal other variances.

        Parameters
        ----------
        Dopt_path : TYPE
            DESCRIPTION.
        ds : TYPE
            DESCRIPTION.
        r : TYPE
            DESCRIPTION.
        n : TYPE
            DESCRIPTION.
        p_empty : TYPE
            DESCRIPTION.
        p_zero : TYPE
            DESCRIPTION.
        alpha_reg : TYPE
            DESCRIPTION.
        locations_to_compare : 

        Returns
        -------
        None.

        """
        
        variances = np.array([1e-6,1e-4,1e-2,1e-1,1e0])
        lowrank_basis = LRB.LowRankBasis(ds, r)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)
        locations = {el:[] for el in variances}
        # Load locations obtained for a given variances ratio
        for var in variances:
            sensor_placement = SP.SensorPlacement('D_optimal', n, r, p_zero,n-(p_zero+p_empty),
                                                  p_empty, var_eps, var)
            
            sensor_placement.initialize_problem(lowrank_basis.Psi,alpha_reg)
            sensor_placement.LoadLocations(Dopt_path, alpha_reg, var)
            
            sensor_placement.locations = sensor_placement.dict_locations[p_zero]
            sensor_placement.weights = sensor_placement.dict_weights[p_zero]
          
            if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
                print(f'No solution found for Dopt with {p_zero} reference stations\n')
                locations[var] = [[],[],[]]
                
            else:
                sensor_placement.C_matrix()
                sensor_placement.covariance_matrix(lowrank_basis.Psi,metric=placement_metric,activate_error_solver=False)
                locations[var] = sensor_placement.locations
            
        df_comparison = pd.DataFrame(index=[f'{i:.1e}' for i in variances],columns=[f'{i:.1e}' for i in variances])
        if locations_to_compare == 'empty':
            idx=2
            num_locations = p_empty
        elif locations_to_compare == 'RefSt':
            idx=1
            num_locations = p_zero
        elif locations_to_compare == 'LCSs':
            idx=0
            num_locations = n-(p_zero+p_empty)
          
       
        for var in variances:
            loc1 = locations[var][idx]
            df_comparison.loc[f'{var:.1e}'][f'{var:.1e}'] = 100. if len(loc1) != 0 else np.nan
            for var2 in np.delete(variances,np.argwhere(variances==var)):
                loc2 = locations[var2][idx]
                df_comparison.loc[f'{var:.1e}'][f'{var2:.1e}'] = 100*np.intersect1d(loc1, loc2).shape[0]/num_locations if (len(loc1) != 0 and len(loc2)!=0) else np.nan
        
        print(f'Percentage of shared locations for {num_locations} {locations_to_compare}\n{df_comparison}')
        return df_comparison
    
    def compare_logdet_convergence(self,Dopt_path,ds,r,n,p_empty,p_zero,var_eps,alpha_reg):
        """
        Load locations obtained with Doptimal for different variances and compute logdet for smaller variances
        comparing with the performance of the Doptimal solutions.

        Parameters
        ----------
        Dopt_path : TYPE
            DESCRIPTION.
        ds : TYPE
            DESCRIPTION.
        r : TYPE
            DESCRIPTION.
        n : TYPE
            DESCRIPTION.
        p_empty : TYPE
            DESCRIPTION.
        p_zero : TYPE
            DESCRIPTION.
        alpha_reg : TYPE
            DESCRIPTION.
        locations_to_compare : 

        Returns
        -------
        None.

        """
        
        variances = np.array([1e-6,1e-4,1e-2,1e-1,1e0])
        lowrank_basis = LRB.LowRankBasis(ds, r)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)
        locations = {el:[] for el in variances}
        logdet = {el:np.inf for el in variances}
        # Load locations obtained for a given variances ratio
        for var in variances:
            sensor_placement = SP.SensorPlacement('D_optimal', n, r, p_zero,n-(p_zero+p_empty),
                                                  p_empty, var_eps, var)
            
            sensor_placement.initialize_problem(lowrank_basis.Psi,alpha_reg)
            sensor_placement.LoadLocations(Dopt_path, alpha_reg, var)
            
            sensor_placement.locations = sensor_placement.dict_locations[p_zero]
            sensor_placement.weights = sensor_placement.dict_weights[p_zero]
          
            if sensor_placement.weights[0].sum() == 0.0 and sensor_placement.weights[1].sum()==0.0:
                print(f'No solution found for Dopt with {p_zero} reference stations\n')
                locations[var] = [[],[],[]]
                logdet[var] = np.inf
                
            else:
                sensor_placement.C_matrix()
                sensor_placement.covariance_matrix(lowrank_basis.Psi,metric=placement_metric,activate_error_solver=False)
                locations[var] = sensor_placement.locations
                logdet[var] = sensor_placement.metric
            
            
        df_convergence = pd.DataFrame(index=[f'{i:.1e}' for i in variances],columns=[f'{i:.1e}' for i in variances])
              
        for var in variances:
            df_convergence.loc[f'{var:.1e}'][f'{var:.1e}'] = logdet[var]
            loc = locations[var]
       
            for var2 in np.delete(variances,np.argwhere(variances==var)):
                sensor_placement = SP.SensorPlacement('D_optimal', n, r, p_zero,n-(p_zero+p_empty),
                                                      p_empty, var_eps, var2)
                sensor_placement.initialize_problem(lowrank_basis.Psi,alpha_reg)
                sensor_placement.locations = loc
                if len(sensor_placement.locations[2]) != 0.0:
                    sensor_placement.C_matrix()
                    sensor_placement.covariance_matrix(lowrank_basis.Psi,metric=placement_metric,activate_error_solver=False)
                    df_convergence.loc[f'{var:.1e}'][f'{var2:.1e}'] = sensor_placement.metric
                else:
                    df_convergence.loc[f'{var:.1e}'][f'{var2:.1e}'] = np.inf
                
        return df_convergence
                
            
            
        
    def show_histograms(self,r,n,p_empty,p_zero_plot,alpha_reg,solving_algorithm,save_fig):
        plots.plot_locations_evolution(self.Dopt_path,self.rank_path,
                                       r,n,p_empty,p_zero_plot,alpha_reg,save_fig=save_fig)
        plots.plot_weights_evolution(Dopt_path,rank_path,r,n,p_empty,solving_algorithm,p_zero_plot,alpha_reg)
        plots.plot_locations_rankMax_alpha(self.rank_path,r,n,p_empty,p_zero_plot,save_fig=save_fig)
        
    def compute_solver_failures(self,r,p_empty,n,var_zero):
        count = 0
        fname = self.Dopt_path+f'Weights_D_optimal_vs_p0_r{r}_pEmpty{p_empty}_varZero{var_zero:.1e}.pkl'
        with open(fname,'rb') as f:
            dict_weights = pickle.load(f)
    
        for p_zero in np.arange(1,n-p_empty+1,1):
            
            weights_lcs = dict_weights[p_zero][0]
            weights_refst = dict_weights[p_zero][1]
            
            if weights_lcs.sum() == 0.0 and weights_refst.sum() == 0.0:
                count+=1
                print(f'Solver failed for {p_zero} reference stations')
                
        print(f'Solver failed for {count} iterations')
        
        

if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
    # dataset to use
    print('Loading data set')
    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    dataset_source = 'synthetic' #['synthetic','real']
    
    dataset = DS.DataSet(pollutant, start_date, end_date, files_path,source=dataset_source)
    dataset.load_dataSet()
    dataset.cleanMissingvalues(strategy='stations',tol=0.1)
    dataset.cleanMissingvalues(strategy='remove')
    
    # train/val/test split
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
    alpha_reg = 1e-1
    num_random_placements = 100
    solving_algorithm = 'D_optimal' #['D_optimal','rankMax']
    placement_metric = 'logdet'

    # lowrank_basis = LRB.LowRankBasis(dataset.ds_train, r)
    # lowrank_basis.snapshots_matrix()
    # lowrank_basis.low_rank_decomposition(normalize=True)    
    # dataset.project_basis(lowrank_basis.Psi)
    
    
    plt.close('all')
    place_sensors = False
    if place_sensors:
        print('Sensor placement\nCompute weights and locations')
        placement = CW.Placement(n, p_empty, r, dataset.ds_train, solving_algorithm, 
                                 var_eps, var_zero, num_random_placements, alpha_reg)
        placement.placement_allStations()
        placement.save_results(results_path)
    else:
        Dopt_path = results_path+'Synthetic_Data/TrainingSet_results/Doptimal/'
        rank_path = results_path+'Synthetic_Data/TrainingSet_results/rankMax/'
        plots = Plots.Plots(save_path=results_path,marker_size=1,fs_label=3,fs_ticks=7,fs_legend=3,fs_title=10,show_plots=True)
        
        plt.close('all')
        p_zero_plot = 40
        
        placement_solutions = PlacementSolutions(Dopt_path, rank_path)
        placement_solutions.compute_solver_failures(r,p_empty,n,1e-6)
        placement_solutions.compute_logdet_configuration(num_random_placements,dataset.ds_train,
                                                         r,n,p_zero_plot,p_empty,save_fig=True)
        placement_solutions.show_histograms(r, n, p_empty, p_zero_plot, 1e-1, solving_algorithm,save_fig=False)
        
        # compare different locations
        df_comparison_empty = placement_solutions.compare_similarities_locations(Dopt_path,dataset.ds_train,
                                                                                 r,n,p_empty,p_zero_plot,var_eps,alpha_reg,
                                                                                 locations_to_compare='empty')
        df_comparison_refst = placement_solutions.compare_similarities_locations(Dopt_path,dataset.ds_train,
                                                                                 r,n,p_empty,p_zero_plot,var_eps,alpha_reg,
                                                                                 locations_to_compare='RefSt')
        df_comparison_lcs= placement_solutions.compare_similarities_locations(Dopt_path,dataset.ds_train,
                                                                              r,n,p_empty,p_zero_plot,var_eps,alpha_reg,
                                                                              locations_to_compare='LCSs')        
        
        df_logdet_convergence = placement_solutions.compare_logdet_convergence(Dopt_path,dataset.ds_train,
                                                                               r,n,p_empty,p_zero_plot,var_eps,alpha_reg)
        
  
        
    validate = False
    if validate:
        print(f'Validation {placement_metric} results on hold-out dataset')
        if dataset_source == 'real':
            train_path = results_path+f'Unmonitored_locations/Training_Testing_split/TrainingSet_results/{solving_algorithm}/'
        else:
            train_path = results_path+f'Synthetic_Data/TrainingSet_results/{solving_algorithm}/'
        
        validation = VT.Validate(n,p_empty,r,alpha_reg,
                              dataset.ds_val, train_path,
                              num_random_placements,
                              var_eps, var_zero,
                              placement_metric,solving_algorithm)
        
        validation.compute_validation()
        validation.save_results(results_path)
    else:
        p_zero_plot = 30
        plots = Plots.Plots(save_path=results_path,marker_size=1,fs_label=3,fs_legend=4,fs_title=10,show_plots=False)
        
        
    estimate = False
    if estimate:
        Dopt_path = results_path+'Synthetic_Data/TrainingSet_results/Doptimal/'
        rank_path = results_path+'Synthetic_Data/TrainingSet_results/rankMax/'
       
        p_zero_estimate = 20
        ds_estimation = dataset.ds_train
        locations_to_estimate='All'
        variances = [1e-6,1e-4,1e-2,1e-1,1e0]
        dict_rmse_var = {el:np.inf for el in variances}
        for var in variances:
            estimation = Estimation.Estimation(n, r, p_empty,p_zero_estimate, var_eps, var, num_random_placements, alpha_reg, 
                                               solving_algorithm, placement_metric, ds_estimation, 
                                               Dopt_path, rank_path, locations_to_estimate)
            
        
        
        
            estimation.random_placement_estimation(rank_path)
            estimation.Dopt_placement_estimation()
            estimation.rankMax_placement_estimation()
            dict_rmse_var[var] = [estimation.rmse_best_random,estimation.rmse_Dopt,estimation.rmse_rankMax]
        
        plots = Plots.Plots(save_path=results_path,marker_size=1,fs_label=3,fs_ticks=8,fs_legend=6,fs_title=10,show_plots=True)
        plots.plot_rmse_vs_variances(dict_rmse_var,p_zero_estimate,p_empty,n,r,
                                     metric='RMSE',locations_estimated=locations_to_estimate,save_fig=True)
        #estimation.compute_estimations()
        #estimation.save_results(results_path)
    
    else:
        # Dopt_path = results_path+'Synthetic_Data/TestingSet_results/Doptimal/'
        # rank_path = results_path+'Synthetic_Data/TestingSet_results/rankMax/'
        
        # Dopt_path = results_path+'Synthetic_Data/TrainingSet_results/Doptimal/'
        # rank_path = results_path+'Synthetic_Data/TrainingSet_results/rankMax/'
        
        
        # plots = Plots.Plots(save_path=results_path,marker_size=1,fs_label=3,fs_ticks=8,fs_legend=6,fs_title=10,show_plots=True)
        # plots.plot_rmse_vs_variances(Dopt_path,rank_path,r,p_empty,p_zero_plot,n,alpha_reg,metric='RMSE',
        #                              locations_to_estimate='empty',save_fig=False)
       
        print('---')
    
  