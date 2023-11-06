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
   
        
    def compare_similarities_locations(self,Dopt_path,ds,r,n,p_empty,p_zero,var_eps,alpha_reg,locations_to_compare='empty'):
        """
        Load locations obtained with Doptimal for different variances and compare the percentage of similarity
        with results with Doptimal for other variances.

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
        
        variances = np.array([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
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
    
    def compare_similarities_rankMax_Dopt(self,Dopt_path,rank_path,alpha_reg,r,p_zero,p_empty,locations_to_compare):
        # load files
        fname = rank_path+f'DiscreteLocations_rankMax_vs_p0_r{r}_pEmpty{p_empty}_alpha{alpha_reg:.1e}.pkl'
        with open(fname,'rb') as f:
            dict_rank_locs = pickle.load(f)
        
        locations_rankMax = dict_rank_locs[p_zero]
        
        variances = np.array([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
        locations_Dopt = {el:[] for el in variances}
        
        for var in variances:
            fname = Dopt_path+f'DiscreteLocations_D_optimal_vs_p0_r{r}_pEmpty{p_empty}_varZero{var:.1e}.pkl'
            with open(fname,'rb') as f:
                dict_Dopt_locs = pickle.load(f)
            if dict_Dopt_locs[p_zero][2].shape[0] == p_empty:
                locations_Dopt[var] = dict_Dopt_locs[p_zero]
            else:
                locations_Dopt[var] = [[],[],[]]
        
        # compute similarities/shared locations assigned to RefSt,LCSs,empty in both cases
        df_comparison = pd.DataFrame(index=[f'{alpha_reg:.1e}'],columns=[f'{i:.1e}' for i in variances])
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
            loc_var = locations_Dopt[var][idx]
            loc_rank = locations_rankMax[idx]
            df_comparison.loc[f'{alpha_reg:.1e}'][f'{var:.1e}'] = 100*np.intersect1d(loc_rank, loc_var).shape[0]/num_locations if (len(locations_rankMax) != 0 and len(loc_var)!=0) else np.nan
        
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
        
        variances = np.array([1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
        lowrank_basis = LRB.LowRankBasis(ds, r)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)
        locations = {el:[] for el in variances}
        weights = {el:[] for el in variances}
        logdet = {el:np.inf for el in variances}
        logdet_convex = {el:np.inf for el in variances}
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
                weights[var] = [[],[]]
                logdet[var] = np.inf
                logdet_convex[var] = np.inf
                
            else:
                sensor_placement.C_matrix()
                sensor_placement.covariance_matrix(lowrank_basis.Psi,metric=placement_metric,activate_error_solver=False)
                locations[var] = sensor_placement.locations
                logdet[var] = sensor_placement.metric
                
                weights[var] = sensor_placement.weights
                sensor_placement.compute_convex_covariance_matrix(lowrank_basis.Psi, sensor_placement.weights,
                                                                                       var, var_eps)
                logdet_convex[var] = sensor_placement.metric_convex
            
            
        df_convergence = pd.DataFrame(index=[f'{i:.1e}' for i in variances],columns=[f'{i:.1e}' for i in variances])
        df_convergence_convex = pd.DataFrame(index=[f'{i:.1e}' for i in variances],columns=[f'{i:.1e}' for i in variances])
              
        for var in variances:
            df_convergence.loc[f'{var:.1e}'][f'{var:.1e}'] = logdet[var]
            df_convergence_convex.loc[f'{var:.1e}'][f'{var:.1e}'] = logdet_convex[var]
            
            loc = locations[var]
            w = weights[var]
       
            for var2 in np.delete(variances,np.argwhere(variances==var)):
                sensor_placement = SP.SensorPlacement('D_optimal', n, r, p_zero,n-(p_zero+p_empty),
                                                      p_empty, var_eps, var2)
                sensor_placement.initialize_problem(lowrank_basis.Psi,alpha_reg)
                sensor_placement.locations = loc
                sensor_placement.weights = w
                
                if len(sensor_placement.locations[2]) != 0.0:
                    sensor_placement.C_matrix()
                    sensor_placement.covariance_matrix(lowrank_basis.Psi,metric=placement_metric,activate_error_solver=False)
                    df_convergence.loc[f'{var:.1e}'][f'{var2:.1e}'] = sensor_placement.metric
                    
                    sensor_placement.compute_convex_covariance_matrix(lowrank_basis.Psi, sensor_placement.weights, sensor_placement.var_zero, sensor_placement.var_eps)
                    df_convergence_convex.loc[f'{var:.1e}'][f'{var2:.1e}'] = sensor_placement.metric
                    
                else:
                    df_convergence.loc[f'{var:.1e}'][f'{var2:.1e}'] = np.inf
                    df_convergence_convex.loc[f'{var:.1e}'][f'{var2:.1e}'] = np.inf
                
        return df_convergence, df_convergence_convex
                
            
            
        
    def show_histograms(self,r,n,p_empty,p_zero_plot,alpha_reg,solving_algorithm,save_fig):
        plots.plot_locations_evolution(self.Dopt_path,self.rank_path,
                                       r,n,p_empty,p_zero_plot,alpha_reg,save_fig=save_fig)
        plots.plot_weights_evolution(Dopt_path,rank_path,r,n,p_empty,solving_algorithm,p_zero_plot,alpha_reg)
        plots.plot_locations_rankMax_alpha(self.rank_path,r,n,p_empty,p_zero_plot,save_fig=save_fig)
        
    def compute_solver_failures(self,r,p_empty,n,var_zero):
        """
        Check for how many locations Doptimal criterion failed to solve problem

        Parameters
        ----------
        r : int
            low-rank subspace dimension
        p_empty : int
            number of unmonitored locaitons
        n : int
            total number of possible locations
        var_zero : float
            variances ratio

        Returns
        -------
        None.

        """
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
                
        print(f'Solver failed for {count} iterations with variances ratio {var_zero}')
        
        

if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
    # dataset to use
    print('Loading data set')
    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    dataset_source = 'cat' #['synthetic','cat','korea]
    
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
    set_sparsity = False
    if set_sparsity:
        r = 1
        percentage_threshold = 0.9
        lowrank_basis = LRB.LowRankBasis(dataset.ds_train, r)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)
        lowrank_basis.cumulative_sum(percentage_threshold)
        plots = Plots.Plots(save_path=results_path,marker_size=1,fs_label=7,fs_ticks=7,fs_legend=5,fs_title=10,show_plots=True)
        plots.plot_singular_values(lowrank_basis.S,dataset_source,save_fig=True)
        
            
        
    # sparsity for 90% (95% synthetic)
    if dataset_source == 'synthetic':
        r = 54
    elif dataset_source == 'cat':
        r = 34
    elif dataset_source == 'korea':
        r = 1
    
    # variances ratio and regularization parameter
    p_empty = n-r
    var_eps,var_zero = 1,1e-6
    alpha_reg = 1e-2
    num_random_placements = int(1e3)
    solving_algorithm = 'D_optimal' #['D_optimal','rankMax']
    placement_metric = 'logdet'
   
    
    plt.close('all')
    place_sensors = False
    if place_sensors:
        print('Sensor placement\nCompute weights and locations')
        placement = CW.Placement(n, p_empty, r, dataset.ds_train, solving_algorithm, 
                                 var_eps, var_zero, num_random_placements, alpha_reg)
        print(f'Sensor placement:\ndata set: {dataset_source}\nNetwork size: {n}\nrandom placements: {num_random_placements}\ncriterion: {solving_algorithm}\nsparsity: {r}\nunmonitored locations: {p_empty}\nalpha for rankMax: {alpha_reg:.1e}\nvariances ratio for Doptimal: lcs {var_eps:.1e}, refst {var_zero:.1e}')
        input('Press Enter to continue ...')
        placement.placement_random()
        placement.placement_allStations()
        placement.save_results(results_path)
        
        
    else:
        # directory where files are stored: depends on dataset
        if dataset_source == 'synthetic':
            Dopt_path = results_path+'Synthetic_Data/TrainingSet_results/Doptimal/'
            rank_path = results_path+'Synthetic_Data/TrainingSet_results/rankMax/'
        elif dataset_source == 'cat':
            Dopt_path = results_path+'Cat/TrainingSet_results/Doptimal/'
            rank_path = results_path+'Cat/TrainingSet_results/rankMax/'
        else:
            Dopt_path = results_path+'Korea/TrainingSet_results/Doptimal/'
            rank_path = results_path+'Korea/TrainingSet_results/rankMax/'
            
        plots = Plots.Plots(save_path=results_path,marker_size=1,fs_label=3,fs_ticks=7,fs_legend=3,fs_title=10,show_plots=False)
        
        plt.close('all')
        p_zero_plot = 10
        alpha_plot = 1e-2 # from validation results for that number of ref.st.
        
        placement_solutions = PlacementSolutions(Dopt_path, rank_path)
        
        # analyze solver failures
        variances = np.logspace(-6,0,7)
        for var in variances:
            placement_solutions.compute_solver_failures(r,p_empty,n,var)
      
        # weights histograms and discrete conversion
        plots.plot_weights_evolution(Dopt_path,rank_path,r,n,p_empty,solving_algorithm,p_zero_plot,alpha_plot)
        
        plots.plot_locations_evolution(Dopt_path,rank_path,r,n,p_empty,p_zero_plot,alpha_plot,save_fig=False)
        
        
        plots.plot_locations_rankMax_alpha(rank_path,r,n,p_empty,p_zero_plot,save_fig=False)
        
        
        # compare shared locations between different Doptimal solutions
        df_logdet_convergence,df_logdet_convergence_convex = placement_solutions.compare_logdet_convergence(Dopt_path,dataset.ds_train,
                                                                               r,n,p_empty,p_zero_plot,var_eps,alpha_reg)
        
        df_comparison_empty = placement_solutions.compare_similarities_locations(Dopt_path,dataset.ds_train,    
                                                                                 r,n,p_empty,p_zero_plot,var_eps,alpha_plot,
                                                                                 locations_to_compare='empty')
        df_comparison_refst = placement_solutions.compare_similarities_locations(Dopt_path,dataset.ds_train,
                                                                                 r,n,p_empty,p_zero_plot,var_eps,alpha_plot,
                                                                                 locations_to_compare='RefSt')
        df_comparison_lcs= placement_solutions.compare_similarities_locations(Dopt_path,dataset.ds_train,
                                                                              r,n,p_empty,p_zero_plot,var_eps,alpha_plot,
                                                                              locations_to_compare='LCSs')    
        
        # compare similarities between rankMax distribution and different Dopt solutions
        
        placement_solutions.compare_similarities_rankMax_Dopt(Dopt_path, rank_path,
                                                              alpha_plot, r, p_zero_plot, p_empty, 
                                                              locations_to_compare='empty')
        
        placement_solutions.compare_similarities_rankMax_Dopt(Dopt_path, rank_path,
                                                              alpha_plot, r, p_zero_plot, p_empty, 
                                                              locations_to_compare='LCSs')
        
        placement_solutions.compare_similarities_rankMax_Dopt(Dopt_path, rank_path,
                                                              alpha_plot, r, p_zero_plot, p_empty, 
                                                              locations_to_compare='RefSt')
      
     
        
       
        
        
        
  
        
    validate = False
    if validate:
        print(f'Validation {placement_metric} results on hold-out dataset')
        if dataset_source == 'synthetic':
            train_path = results_path+f'Synthetic_Data/TrainingSet_results/rankMax/'
        elif dataset_source == 'cat':
            train_path = results_path+f'Cat/TrainingSet_results/rankMax/'
        else:
            train_path = results_path+f'Korea/TrainingSet_results/rankMax/'
        
            
        lowrank_basis = LRB.LowRankBasis(dataset.ds_train, r)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)    
        dataset.project_basis(lowrank_basis.Psi)
      
        p_zero_estimate = 30
        locations_to_estimate='All'
        
        alphas = np.logspace(-2,2,5)
        
        var_zero = 0.0 #rankMax gives solution for exact case variance == 0
        
        ds_lcs_train = dataset.perturbate_signal(dataset.ds_train_projected, 15*var_eps, seed=92)
        ds_refst_train = dataset.perturbate_signal(dataset.ds_train_projected, 15*var_zero, seed=92)
        ds_real_train = dataset.ds_train_projected
        
        ds_lcs_val = dataset.perturbate_signal(dataset.ds_val_projected, 15*var_eps, seed=92)
        ds_refst_val = dataset.perturbate_signal(dataset.ds_val_projected, 15*var_zero, seed=92)
        ds_real_val = dataset.ds_val_projected
        
        rmse_alpha_train = {el:[] for el in alphas}
        rmse_alpha_val = {el:[] for el in alphas}
        for alpha_reg in alphas:
            print(f'Regularization for alpha: {alpha_reg}')
            estimation_train = Estimation.Estimation(n, r, p_empty, p_zero_estimate, var_eps, var_zero, 
                                                     num_random_placements, alpha_reg,
                                                     solving_algorithm, placement_metric, 
                                                     ds_lcs_train, ds_refst_train, ds_real_train, 
                                                     lowrank_basis.Psi, Dopt_path, rank_path, locations_to_estimate)
            
            estimation_train.rankMax_placement_estimation_limit()
            
            rmse_alpha_train[alpha_reg] = estimation_train.rmse_rankMax
            
            
            estimation_val = Estimation.Estimation(n, r, p_empty, p_zero_estimate, var_eps, var_zero, 
                                                     num_random_placements, alpha_reg,
                                                     solving_algorithm, placement_metric, 
                                                     ds_lcs_val, ds_refst_val, ds_real_val, 
                                                     lowrank_basis.Psi, Dopt_path, rank_path, locations_to_estimate)
            
            estimation_val.rankMax_placement_estimation_limit()
            
            
            rmse_alpha_val[alpha_reg] = estimation_val.rmse_rankMax
            
        print(f'Training set results for epsilon = 0\n{rmse_alpha_train}\nValidation results\n{rmse_alpha_val}')
        fname = results_path+f'RMSE_{locations_to_estimate}_validation_{p_zero_estimate}RefSt.pkl'
        with open(fname,'wb') as f:
           pickle.dump(rmse_alpha_val,f)
        fname = results_path+f'RMSE_{locations_to_estimate}_train_{p_zero_estimate}RefSt.pkl'
        with open(fname,'wb') as f:
           pickle.dump(rmse_alpha_train,f)
           
    
     
    else:
        if dataset_source == 'synthetic':
            val_path = results_path+'Synthetic_Data/ValidationSet_results/'
        elif dataset_source == 'cat':
            val_path = results_path+'Cat/ValidationSet_results/'
        else:
            val_path = results_path+'Korea/ValidationSet_results/'
            
        plt.close('all')
        plots = Plots.Plots(save_path=results_path,marker_size=1,fs_label=10,fs_ticks=7,fs_legend=7,fs_title=10,show_plots=True)
        plots.plot_rmse_validation(val_path,p_zero_plot=30,save_fig=True)
        
        print('Validation Finished')
        
    estimate = True
    if estimate:
        
        print('Estimation on testing set.\nComparing Doptimal solutions with rankMax solutions using GLS estimations.')
        input('Press Enter to continue ...')
        
        if dataset_source == 'synthetic':
            Dopt_path = results_path+'Synthetic_Data/TrainingSet_results/Doptimal/'
            rank_path = results_path+'Synthetic_Data/TrainingSet_results/rankMax/'
            p_zero_range_validated = [10,20,30,40,50]
            alphas_range = [1e2,1e-2,1e1,1e0,1e-2]
            alphas = {el:a for el,a in zip(p_zero_range_validated,alphas_range)}
            
        elif dataset_source == 'cat':
            Dopt_path = results_path+'Cat/TrainingSet_results/Doptimal/'
            rank_path = results_path+'Cat/TrainingSet_results/rankMax/'
            p_zero_range_validated = [10,20,30]
            alphas_range = [1e-2,1e-2,1e-2]
            alphas = {el:a for el,a in zip(p_zero_range_validated,alphas_range)}
            
        else:
            Dopt_path = results_path+'Korea/TrainingSet_results/Doptimal/'
            rank_path = results_path+'Korea/TrainingSet_results/rankMax/'
            p_zero_range_validated = []
            alphas_range = []
            alphas = {el:a for el,a in zip(p_zero_range_validated,alphas_range)}
        
        # set of validated number of reference stations and respective alphas for rankMax criterion (on whole network)
        
        
        # estimate RMSE in the whole network using Dopt(variance dependant) and rankMax(alpha dependant) configurations.
        # For a given variance the estimations are computed using GLS for both criteria
        # rankMax gives an extra estimation for variance == 0.
        
        lowrank_basis = LRB.LowRankBasis(dataset.ds_train, r)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)
        dataset.project_basis(lowrank_basis.Psi)
        
        p_zero_estimate = 10
        alpha_reg = alphas[p_zero_estimate]
        locations_to_estimate='All'
        
        variances = np.concatenate(([0.0],np.logspace(-6,0,7)))
        dict_rmse_var = {el:np.inf for el in variances}
        
        for var in variances:
            
            ds_lcs_test = dataset.perturbate_signal(dataset.ds_test_projected, var_eps, seed=92)#30/10 synthetic (var -2..0/-3..-6)/20 cat
            ds_refst_test = dataset.perturbate_signal(dataset.ds_test_projected,var, seed=92)
            ds_real_test = dataset.ds_test_projected
        
            
            estimation_test = Estimation.Estimation(n, r, p_empty, p_zero_estimate, var_eps, var, 
                                                     num_random_placements, alpha_reg,
                                                     solving_algorithm, placement_metric, 
                                                     ds_lcs_test, ds_refst_test, ds_real_test, 
                                                     lowrank_basis.Psi, Dopt_path, rank_path, locations_to_estimate)
        
        
            if var!=0:
                estimation_test.random_placement_estimation(rank_path)
                estimation_test.Dopt_placement_estimation()
                estimation_test.Dopt_placement_convergene(var_orig=1e0)
                estimation_test.rankMax_placement_estimation()
                dict_rmse_var[var] = [estimation_test.rmse_ci_random,estimation_test.rmse_Dopt,estimation_test.rmse_Dopt_convergence,estimation_test.rmse_rankMax]
                
            else:
                estimation_test.random_placement_estimation_limit(rank_path)
                estimation_test.Dopt_placement_convergene(var_orig=1e0)
                estimation_test.rankMax_placement_estimation_limit()
                dict_rmse_var[var] = [estimation_test.rmse_ci_random,np.inf,estimation_test.rmse_Dopt_convergence,estimation_test.rmse_rankMax]
            
            
        
        
        fname = f'{results_path}RMSE_vs_variances_{locations_to_estimate}_RandomDoptRankMax_{p_zero_estimate}RefSt.pkl'
        with open(fname,'wb') as f:
           pickle.dump(dict_rmse_var,f)
        
      
    else:
        # if dataset_source == 'real':
        #     test_path = results_path+f'Unmonitored_locations/TestingSet_results/'
        # else:
        #     test_path = results_path+f'Synthetic_Data/TestingSet_results/'
         
        # plt.close('all')
      
        # p_zero_range_validated = [10,20,30,40,50]
        # if dataset_source == 'synthetic':
        #     alphas_range = [1e2,1e-2,1e1,1e0,1e-2]
        #     alphas = {el:a for el,a in zip(p_zero_range_validated,alphas_range)}
        # else:
        #     print('regularization parameter alpha not validated in the limit')
        #     alphas_range = [1e2,1e-2,1e1,1e0,1e-2]
       
        
        # p_zero_plot = 10
        # alpha_reg = alphas[p_zero_plot]
        # placement_solutions = PlacementSolutions(Dopt_path, rank_path)
        # placement_solutions.compare_similarities_rankMax_Dopt(Dopt_path, rank_path,
        #                                                       alpha_reg, r, p_zero_plot, p_empty, 
        #                                                       locations_to_compare='empty')
        
        # placement_solutions.compare_similarities_rankMax_Dopt(Dopt_path, rank_path,
        #                                                       alpha_reg, r, p_zero_plot, p_empty, 
        #                                                       locations_to_compare='LCSs')
        
        # placement_solutions.compare_similarities_rankMax_Dopt(Dopt_path, rank_path,
        #                                                      alpha_reg, r, p_zero_plot, p_empty, 
        #                                                      locations_to_compare='RefSt')
      
     
        
        # plots = Plots.Plots(save_path=results_path,marker_size=1,fs_label=7,fs_ticks=7,fs_legend=5,fs_title=10,show_plots=True)
        # plots.plot_rmse_vs_variances(test_path,p_zero_plot,alpha_reg,locations_estimated='All',save_fig=True)
       
        # plots.plot_execution_time_variance(Dopt_path,rank_path,r,p_empty,p_zero_plot,alpha_reg,save_fig=True)
     
        print('--')
        
    
  