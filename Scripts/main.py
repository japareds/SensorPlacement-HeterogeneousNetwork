#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 13:42:32 2023

@author: jparedes
"""


import os
import sys
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
        
class EstimationPhase():
    def __init__(self,n,p_zero_range,p_empty,r,num_random_placements,Dopt_path,rank_path,alphas,var_eps,save_path):
        self.n = n
        self.n_refst_range = p_zero_range
        self.n_unmonitored = p_empty
        self.sparsity = r
        self.num_random_placements = num_random_placements
        self.Dopt_path = Dopt_path
        self.rank_path = rank_path
        self.alphas = alphas
        self.var_lcs = var_eps
        self.save_path = save_path
        
    def compute_analytical_errors(self,dataset,save_results=False):
        """
        Compute analytical RMSE from trace covariance matrix for multiple number of reference stations in the network and 
        different variances ratio between refst and LCSs.
        
        Given a number of reference stations (and LCSs) and variances ratio:
            1) Locations are loaded for both criteria
            2) regressor and measurement covariance matrices are computed and the metric measured
            3) The results are stored for both criteria and at different locations.

        Parameters
        ----------
        dataset : DataSet object
            original dataset with measurements
        save_results : boolean
            save generated dictionaries

        Returns
        -------
        None.

        """
        input('Compute analytical RMSE using covariance matrices.\nPress Enter to continue ...')
        # project dataset onto subspace so that data is exactly sparse in a basis
        lowrank_basis = LRB.LowRankBasis(dataset.ds_train,self.sparsity)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)
        dataset.project_basis(lowrank_basis.Psi)
        
        variances = np.concatenate(([0.0],np.logspace(-6,0,7)))
        
        self.dict_rmse_refst_Dopt = {el:0 for el in self.n_refst_range}
        self.dict_rmse_refst_rank = {el:0 for el in self.n_refst_range}
        
        for n_refst in self.n_refst_range:
            alpha_reg = self.alphas[n_refst]
            
            dict_Dopt_var = {el:np.inf for el in variances}
            dict_rankMax_var = {el:np.inf for el in variances}
            for var in variances:
                ds_lcs_test = dataset.perturbate_signal(dataset.ds_test_projected, var_eps, seed=0).copy()
                ds_refst_test = dataset.perturbate_signal(dataset.ds_test_projected,var, seed=0).copy()
                ds_real_test = dataset.ds_test_projected.copy()
                
                # estimate using those generated sensors measurements
                
                estimation_Dopt = Estimation.Estimation(self.n, self.sparsity, self.n_unmonitored, 
                                                        n_refst, self.var_lcs, var, 
                                                        self.num_random_placements, alpha_reg,
                                                        ds_lcs_test, ds_refst_test, ds_real_test,
                                                        lowrank_basis.Psi, self.Dopt_path, self.rank_path)
                
                estimation_rankMax = Estimation.Estimation(self.n, self.sparsity, self.n_unmonitored, 
                                                        n_refst, self.var_lcs, var, 
                                                        self.num_random_placements, alpha_reg,
                                                        ds_lcs_test, ds_refst_test, ds_real_test,
                                                        lowrank_basis.Psi, self.Dopt_path, self.rank_path)
                if var!= 0:
                
                    estimation_Dopt.analytical_estimation(criterion='D_optimal')
                    dict_Dopt_var[var] = [estimation_Dopt.rmse_analytical_full,estimation_Dopt.rmse_analytical_refst,estimation_Dopt.rmse_analytical_lcs,estimation_Dopt.rmse_analytical_unmonitored]
                    estimation_rankMax.analytical_estimation(criterion='rankMax')
                    dict_rankMax_var[var] = [estimation_rankMax.rmse_analytical_full,estimation_rankMax.rmse_analytical_refst,estimation_rankMax.rmse_analytical_lcs,estimation_rankMax.rmse_analytical_unmonitored]
                    
                else:
                    dict_Dopt_var[var] = [np.inf,np.inf,np.inf,np.inf]
                    estimation_rankMax.analytical_estimation(criterion='rankMax')
                    dict_rankMax_var[var] = [estimation_rankMax.rmse_analytical_full,estimation_rankMax.rmse_analytical_refst,estimation_rankMax.rmse_analytical_lcs,estimation_rankMax.rmse_analytical_unmonitored]
                
            self.dict_rmse_refst_Dopt[n_refst] = pd.DataFrame(dict_Dopt_var,index=['Full','RefSt','LCS','Unmonitored'],columns=variances).T
            self.dict_rmse_refst_rank[n_refst] = pd.DataFrame(dict_rankMax_var,index=['Full','RefSt','LCS','Unmonitored'],columns=variances).T
        
        if save_results:
            fname = f'{self.save_path}RMSE_analytical_Doptimal.pkl'
            with open(fname,'wb') as f:
               pickle.dump(self.dict_rmse_refst_Dopt,f)
            
            fname = f'{self.save_path}RMSE_analytical_rankMax.pkl'
            with open(fname,'wb') as f:
               pickle.dump(self.dict_rmse_refst_rank,f)
               
            print(f'results saved in {self.save_path}')
            
            
        
    def compute_measurements_errors(self,dataset,num_refst=10,save_results=False):
        """
        Compute RMSE between real dataset and estimations using network configurationobtained with specific criterion.
        
        Given a certain variances ratio and criterion solution:
        1) The original dataset is perturbated to generate refst and LCS measurements. The perturbation is for a specific seed
        2) The criterion locations are loaded and estimated regressor computed
        3) The estimated measurements are computed and rmse calculated
        4) The output differentiates between location status: monitored by refst, monitored by LCS or unmonitored
        5) The process is repeated for new dataset measurements (seed perturbation).
        6) The process is repeated for different variances ratio.

        Parameters
        ----------
        dataset : dataSet object
            dataset with actual measurements
            
        num_refst : int, optional
            Number of reference stations. The default is 10.
        save_results : bool, optional
            save generated dictionaries. The default is False.

        Returns
        -------
        None.

        """
        # dataset used for estimation and number of reference stations in the network
        
        print(f'Estimation on testing set dataset {dataset_source}.\npollutant {pollutant}.\n Placing {num_refst} reference stations.\nComparing Doptimal solutions with rankMax solutions using GLS estimations.')
        input('Press Enter to continue ...')
        
        lowrank_basis = LRB.LowRankBasis(dataset.ds_train,self.sparsity)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)
        dataset.project_basis(lowrank_basis.Psi)
        
        
        alpha_reg = self.alphas[num_refst]
        
        variances = np.concatenate(([0.0],np.logspace(-6,0,7)))
        seeds = np.arange(0,1e1,1)
        
        # dictionaries for saving results (3 methods)
        dict_random_var = {el:np.inf for el in variances}
        dict_Dopt_var = {el:np.inf for el in variances}
        dict_rankMax_var = {el:np.inf for el in variances}
        
        
        for var in variances: # certain variances ratio
            
            dict_rmse_random = {el:np.inf for el in seeds}
            dict_rmse_Dopt = {el:np.inf for el in seeds}
            dict_rmse_rankMax = {el:np.inf for el in seeds}
            
            for s in seeds: # certain seed for perturbating
                
                # perturbate dataset to generate LCSs(variance var_eps) and ref.st.(variance var_zero). repeat for many seeds
                ds_lcs_test = dataset.perturbate_signal(dataset.ds_test_projected, self.var_lcs, seed=int(s)).copy()
                ds_refst_test = dataset.perturbate_signal(dataset.ds_test_projected,var, seed=int(s)).copy()
                ds_real_test = dataset.ds_test_projected.copy()
                
                # estimate using those generated sensors measurements
                estimation_test = Estimation.Estimation(self.n, self.sparsity,
                                                        self.n_unmonitored, num_refst,
                                                        self.var_lcs, var, 
                                                        self.num_random_placements, alpha_reg,
                                                        ds_lcs_test, ds_refst_test, ds_real_test,lowrank_basis.Psi, 
                                                        self.Dopt_path,self.rank_path)
            
                # case variance ref_st is not zero
                if var!=0:
                    estimation_test.random_placement_estimation(self.rank_path)
                    estimation_test.Dopt_placement_estimation()
                    estimation_test.rankMax_placement_estimation()
                    
                    dict_rmse_random[s] = [estimation_test.rmse_random_full,estimation_test.rmse_random_refst,estimation_test.rmse_random_lcs,estimation_test.rmse_random_unmonitored] 
                    dict_rmse_Dopt[s] = [estimation_test.rmse_Dopt_full,estimation_test.rmse_Dopt_refst,estimation_test.rmse_Dopt_lcs,estimation_test.rmse_Dopt_unmonitored]
                    dict_rmse_rankMax[s] = [estimation_test.rmse_rankMax_full,estimation_test.rmse_rankMax_refst,estimation_test.rmse_rankMax_lcs,estimation_test.rmse_rankMax_unmonitored]
                    
                # case variance ref_st == 0
                else:
                    estimation_test.random_placement_estimation_limit(rank_path)
                    estimation_test.rankMax_placement_estimation_limit()
                    
                    dict_rmse_random[s] = [estimation_test.rmse_random_full,estimation_test.rmse_random_refst,estimation_test.rmse_random_lcs,estimation_test.rmse_random_unmonitored] 
                    dict_rmse_Dopt[s] = [np.inf,np.inf,np.inf,np.inf]
                    dict_rmse_rankMax[s] = [estimation_test.rmse_rankMax_full,estimation_test.rmse_rankMax_refst,estimation_test.rmse_rankMax_lcs,estimation_test.rmse_rankMax_unmonitored]
                    
                    
            # sort seeds iterations as dataframe and store it for the specific variances ratio
            dict_random_var[var] = pd.DataFrame(dict_rmse_random,index=['Full','RefSt','LCS','Unmonitored']).T
            dict_Dopt_var[var] = pd.DataFrame(dict_rmse_Dopt,index=['Full','RefSt','LCS','Unmonitored']).T
            dict_rankMax_var[var] = pd.DataFrame(dict_rmse_rankMax,index=['Full','RefSt','LCS','Unmonitored']).T
            
        
        if save_results:
            fname = f'{self.save_path}RMSE_vs_variances_Random_{num_refst}RefSt.pkl'
            with open(fname,'wb') as f:
                pickle.dump(dict_random_var,f)
        
            fname = f'{self.save_path}RMSE_vs_variances_Dopt_{num_refst}RefSt.pkl'
            with open(fname,'wb') as f:
               pickle.dump(dict_Dopt_var,f)
            
            fname = f'{self.save_path}RMSE_vs_variances_rankMax_{num_refst}RefSt.pkl'
            with open(fname,'wb') as f:
               pickle.dump(dict_rankMax_var,f)
            
            print(f'results saved in {self.save_path}')
            
                
        
        
        
    
        
        
    
        
    
        
#%%

if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
    # dataset to use
    print('Loading data set')
    pollutant = 'O3' #['O3','NO2']
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    dataset_source = 'taiwan' #['synthetic','cat','taiwan]
    
    dataset = DS.DataSet(pollutant, start_date, end_date, files_path,source=dataset_source)
    dataset.load_dataSet()
    if dataset_source != 'taiwan':
        dataset.cleanMissingvalues(strategy='stations',tol=0.1)
    dataset.cleanMissingvalues(strategy='remove')
    
    # train/val/test split
    train_set_end = '2020-12-31 23:00:00' if dataset_source != 'taiwan' else '2021-12-31 23:00:00'
    val_set_begin = '2021-01-01 00:00:00' if dataset_source != 'taiwan' else '2022-01-01 00:00:00'
    val_set_end = '2021-12-31 23:00:00' if dataset_source != 'taiwan' else '2022-12-31 23:00:00'
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
        plots.plot_singular_values(lowrank_basis.S,dataset_source,save_fig=False)
        #plots.plot_sing_val_two_datasets(S_O3, S_NO2, r'$O_3$',r'$NO_2$',save_fig=True)
        
            
        
    # sparsity for 90% (95% synthetic)
    if dataset_source == 'synthetic':
        r = 54
    elif dataset_source == 'cat':
        r = 34 
    elif dataset_source == 'taiwan':
        r = 50 if pollutant == 'O3' else 53
    
    # variances ratio and regularization parameter
    p_empty = n-r
    var_eps,var_zero = 1,1e0
    alpha_reg = -1e-2
    num_random_placements = int(1e3)
    solving_algorithm = 'rankMax' #['D_optimal','rankMax','Dopt-Liu']
    placement_metric = 'logdet'
   
    
    plt.close('all')
    place_sensors = False
    if place_sensors:
        print('Sensor placement\nCompute weights and locations')
        placement = CW.Placement(n, p_empty, r, dataset.ds_train, solving_algorithm, 
                                 var_eps, var_zero, num_random_placements, alpha_reg)
        print(f'Sensor placement:\ndata set: {dataset_source}\npollutant: {pollutant}\nNetwork size: {n}\nrandom placements: {num_random_placements}\ncriterion: {solving_algorithm}\nsparsity: {r}\nunmonitored locations: {p_empty}\nalpha for rankMax: {alpha_reg:.1e}\nvariances ratio for Doptimal: lcs {var_eps:.1e}, refst {var_zero:.1e}')
        input('Press Enter to continue ...')
        placement.placement_random()
        placement.placement_allStations()
        placement.save_results(results_path)
        sys.exit()
        
        
    else:
        # directory where files are stored: depends on dataset
        if dataset_source == 'synthetic':
            Dopt_path = results_path+'Synthetic_Data/TrainingSet_results/Doptimal/'
            rank_path = results_path+'Synthetic_Data/TrainingSet_results/rankMax/'
        elif dataset_source == 'cat':
            Dopt_path = results_path+'Cat/TrainingSet_results/Doptimal/'
            rank_path = results_path+'Cat/TrainingSet_results/rankMax/'
        else:
            Dopt_path = results_path+'Taiwan/TrainingSet_results/Doptimal/' if pollutant == 'O3' else results_path+'Taiwan_NO2/TrainingSet_results/Doptimal/'
            rank_path = results_path+'Taiwan/TrainingSet_results/rankMax/' if pollutant == 'O3' else results_path+'Taiwan_NO2/TrainingSet_results/rankMax/'
            
        plots = Plots.Plots(save_path=results_path,marker_size=1,fs_label=3,fs_ticks=7,fs_legend=3,fs_title=10,show_plots=False)
        
        plt.close('all')
        p_zero_plot = 20
        alpha_plot = 1e-2 # from validation results for that number of ref.st.
        
        placement_solutions = PlacementSolutions(Dopt_path, rank_path)
        
        # analyze solver failures
        variances = np.logspace(-6,0,7)
        for var in variances:
            placement_solutions.compute_solver_failures(r,p_empty,n,var)
      
        # # weights histograms and discrete conversion
        # plots.plot_weights_evolution(Dopt_path,rank_path,r,n,p_empty,solving_algorithm,p_zero_plot,alpha_plot)
        
        # plots.plot_locations_evolution(Dopt_path,rank_path,r,n,p_empty,p_zero_plot,alpha_plot,save_fig=False)
        
        
        # plots.plot_locations_rankMax_alpha(rank_path,r,n,p_empty,p_zero_plot,save_fig=False)
        
        
        # # compare shared locations between different Doptimal solutions
        # df_logdet_convergence,df_logdet_convergence_convex = placement_solutions.compare_logdet_convergence(Dopt_path,dataset.ds_train,
        #                                                                        r,n,p_empty,p_zero_plot,var_eps,alpha_reg)
        
        # df_comparison_empty = placement_solutions.compare_similarities_locations(Dopt_path,dataset.ds_train,    
        #                                                                          r,n,p_empty,p_zero_plot,var_eps,alpha_plot,
        #                                                                          locations_to_compare='empty')
        # df_comparison_refst = placement_solutions.compare_similarities_locations(Dopt_path,dataset.ds_train,
        #                                                                          r,n,p_empty,p_zero_plot,var_eps,alpha_plot,
        #                                                                          locations_to_compare='RefSt')
        # df_comparison_lcs= placement_solutions.compare_similarities_locations(Dopt_path,dataset.ds_train,
        #                                                                       r,n,p_empty,p_zero_plot,var_eps,alpha_plot,
        #                                                                       locations_to_compare='LCSs')    
        
        # # compare similarities between rankMax distribution and different Dopt solutions
        
        # placement_solutions.compare_similarities_rankMax_Dopt(Dopt_path, rank_path,
        #                                                       alpha_plot, r, p_zero_plot, p_empty, 
        #                                                       locations_to_compare='empty')
        
        # placement_solutions.compare_similarities_rankMax_Dopt(Dopt_path, rank_path,
        #                                                       alpha_plot, r, p_zero_plot, p_empty, 
        #                                                       locations_to_compare='LCSs')
        
        # placement_solutions.compare_similarities_rankMax_Dopt(Dopt_path, rank_path,
        #                                                       alpha_plot, r, p_zero_plot, p_empty, 
        #                                                       locations_to_compare='RefSt')
      
        
        
  
        
  
        
    validate = False
    if validate:
        print(f'Validation {placement_metric} results on hold-out dataset')
        input('Press Enter to continue')
        if dataset_source == 'synthetic':
            Dopt_path = results_path+'Synthetic_Data/TrainingSet_results/Doptimal/'
            rank_path = results_path+'Synthetic_Data/TrainingSet_results/rankMax/'
        elif dataset_source == 'cat':
            Dopt_path = results_path+'Cat/TrainingSet_results/Doptimal/'
            rank_path = results_path+'Cat/TrainingSet_results/rankMax/'
        else:
            Dopt_path = results_path+'Taiwan/TrainingSet_results/Doptimal/' if pollutant == 'O3' else results_path+'Taiwan_NO2/TrainingSet_results/Doptimal/'
            rank_path = results_path+'Taiwan/TrainingSet_results/rankMax/' if pollutant == 'O3' else results_path+'Taiwan_NO2/TrainingSet_results/rankMax/'
        
            
        lowrank_basis = LRB.LowRankBasis(dataset.ds_train, r)
        lowrank_basis.snapshots_matrix()
        lowrank_basis.low_rank_decomposition(normalize=True)    
        dataset.project_basis(lowrank_basis.Psi)
      
        p_zero_estimate = 40
        locations_to_estimate='All'
        
        alphas = np.concatenate((-np.logspace(-2,2,5),np.logspace(-2,2,5)))#np.logspace(-2,2,5)
        
        var_zero = 0.0 #rankMax gives solution for exact case variance == 0
        
        ds_lcs_train = dataset.perturbate_signal(dataset.ds_train_projected, var_eps, seed=92)
        ds_refst_train = dataset.perturbate_signal(dataset.ds_train_projected, var_zero, seed=92)
        ds_real_train = dataset.ds_train_projected
        
        ds_lcs_val = dataset.perturbate_signal(dataset.ds_val_projected, var_eps, seed=92)
        ds_refst_val = dataset.perturbate_signal(dataset.ds_val_projected, var_zero, seed=92)
        ds_real_val = dataset.ds_val_projected
        
        rmse_alpha_train = {el:[] for el in alphas}
        rmse_alpha_val = {el:[] for el in alphas}
        for alpha_reg in alphas:
            print(f'Regularization for alpha: {alpha_reg}')
            estimation_train = Estimation.Estimation(n, r, p_empty, p_zero_estimate, var_eps, var_zero, 
                                                     num_random_placements, alpha_reg,
                                                     solving_algorithm, 
                                                     ds_lcs_train, ds_refst_train, ds_real_train, 
                                                     lowrank_basis.Psi, Dopt_path, rank_path, locations_to_estimate)
            
            estimation_train.rankMax_placement_estimation_limit()
            
            rmse_alpha_train[alpha_reg] = estimation_train.rmse_rankMax
            
            
            estimation_val = Estimation.Estimation(n, r, p_empty, p_zero_estimate, var_eps, var_zero, 
                                                     num_random_placements, alpha_reg,
                                                     solving_algorithm, 
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
           
        sys.exit()
     
    else:
        if dataset_source == 'synthetic':
            val_path = results_path+'Synthetic_Data/ValidationSet_results/'
        elif dataset_source == 'cat':
            val_path = results_path+'Cat/ValidationSet_results/'
        else:
            val_path = results_path+'Korea/ValidationSet_results/'
            
        # plt.close('all')
        # plots = Plots.Plots(save_path=results_path,marker_size=1,fs_label=10,fs_ticks=7,fs_legend=7,fs_title=10,show_plots=False)
        # plots.plot_rmse_validation(val_path,p_zero_plot=30,save_fig=False)
        
        print('Validation Finished')
        
    estimate = True
    if estimate:
        
        # estimate RMSE in the whole network using Dopt(variance dependant) and rankMax(alpha dependant) configurations.
        # For a given variance the estimations are computed using GLS for both criteria
        # rankMax gives an extra estimation for variance == 0.
        
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
            alphas_range = [-1e-2,-1e-2,-1e-2]
            alphas = {el:a for el,a in zip(p_zero_range_validated,alphas_range)}
            
        else:
            Dopt_path = results_path+'Taiwan/TrainingSet_results/Doptimal/' if pollutant == 'O3' else results_path+'Taiwan_NO2/TrainingSet_results/Doptimal/'
            rank_path = results_path+'Taiwan/TrainingSet_results/rankMax/' if pollutant == 'O3' else results_path+'Taiwan_NO2/TrainingSet_results/rankMax/'
            p_zero_range_validated = [10,20,25,30,40]
            alphas_range = [1e-2,1e1,-1e-1,-1e1,-1e-2] if pollutant == 'O3' else [-1e0,-1e0,-1e0,-1e2,-1e0]
            alphas = {el:a for el,a in zip(p_zero_range_validated,alphas_range)}
        
        # dataset used for estimation and number of reference stations in the network
        estimation_phase = EstimationPhase(n, p_zero_range_validated, p_empty,
                                           r, num_random_placements,
                                           Dopt_path, rank_path, 
                                           alphas, var_eps, results_path)
        
        print(f'Estimation on testing set dataset {dataset_source}.\npollutant {pollutant}. ')
        
        estimation_phase.compute_analytical_errors(dataset,save_results=False)
        sys.exit()
        
        estimation_phase.compute_measurements_errors(dataset,num_refst=20,save_results=True)
        sys.exit()
        
       
        
        
       
        
      
    else:
        if dataset_source == 'synthetic':
            test_path = results_path+'Synthetic_Data/TestingSet_results/'
            p_zero_range_validated = [10,20,30,40,50]
            alphas_range = [1e2,1e-2,1e1,1e0,1e-2]
            alphas = {el:a for el,a in zip(p_zero_range_validated,alphas_range)}
            
        elif dataset_source == 'cat':
            test_path = results_path+'Cat/TestingSet_results/'
            p_zero_range_validated = [10,20,30]
            alphas_range = [-1e-2,-1e-2,-1e-2]
            alphas = {el:a for el,a in zip(p_zero_range_validated,alphas_range)}
        else:
            test_path = results_path+'Taiwan/TestingSet_results/'
            p_zero_range_validated = [1,10,20,30,40,49]
            alphas_range = [1e-2,1e-2,1e1,1e0,1e2,1e2]
            alphas = {el:a for el,a in zip(p_zero_range_validated,alphas_range)}
            
            
        plt.close('all')
      
        
        
        p_zero_plot = 10
        alpha_reg = alphas[p_zero_plot]
        placement_solutions = PlacementSolutions(Dopt_path, rank_path)
        
        # compare similarities between rankMax criterion and Dopt for different var ratios
        placement_solutions.compare_similarities_rankMax_Dopt(Dopt_path, rank_path,
                                                              alpha_reg, r, p_zero_plot, p_empty, 
                                                              locations_to_compare='empty')
        
        placement_solutions.compare_similarities_rankMax_Dopt(Dopt_path, rank_path,
                                                              alpha_reg, r, p_zero_plot, p_empty, 
                                                              locations_to_compare='LCSs')
        
        placement_solutions.compare_similarities_rankMax_Dopt(Dopt_path, rank_path,
                                                              alpha_reg, r, p_zero_plot, p_empty, 
                                                              locations_to_compare='RefSt')
      
     
        
        plots = Plots.Plots(save_path=results_path,marker_size=1,fs_label=7,fs_ticks=7,fs_legend=5,fs_title=10,show_plots=True)
        plots.plot_rmse_vs_variances(test_path,p_zero_plot,alpha_reg,n,'All',save_fig=True)
       
        plots.plot_execution_time_variance(Dopt_path,rank_path,r,p_empty,p_zero_plot,alpha_reg,save_fig=False)
     
        print('--')
        
    
  