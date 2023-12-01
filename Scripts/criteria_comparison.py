#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:09:41 2023

@author: jparedes
"""

import os
import pandas as pd
import numpy as np
import math
import pickle
import sys
import warnings

import SensorPlacement as SP
import DataSet as DS
import LowRankBasis as LRB
import compute_weights as CW
import Estimation
import Plots
#%% 
# =============================================================================
# Dataset and network
# =============================================================================
  
def check_network_parameters(n,n_refst,n_lcs,n_empty,s):
    if n_refst + n_lcs + n_empty !=n:
        raise Exception('Number of sensors and unmonitored locations does not match network size')
    if n_refst < 0 or n_lcs <0 or n_empty < 0:
        raise ValueError('Negative number of sensors')
    if n_refst >= s:
        warnings.warn(f'Number of reference stations {n_refst} larger than sparsity {s}')
    if n_refst + n_lcs < s:
        raise ValueError(f'Total number of sensors {n_lcs + n_refst} lower than sparsity {s}')
    
    
def load_dataset(files_path,n,pollutant):
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    dataset_source = 'taiwan' #['synthetic','cat','taiwan]
    
    dataset = DS.DataSet(pollutant, start_date, end_date, files_path,dataset_source)
    dataset.load_dataSet()
    dataset.cleanMissingvalues(strategy='remove')
    # shrinking network
    dataset.ds = dataset.ds.iloc[:,:n]
    
    train_set_end = '2021-12-31 23:00:00'
    val_set_begin = '2022-01-01 00:00:00'
    val_set_end = '2022-12-31 23:00:00'
    test_set_begin = '2022-01-01 00:00:00'
    dataset.train_test_split(train_set_end, val_set_begin, val_set_end,test_set_begin)
    
    return dataset

# =============================================================================
# Sensor placement
# =============================================================================

def solve_sensor_placement(dataset,n,n_empty,s,results_path,criterion='rankMax'):
    alphas = np.concatenate((-np.logspace(-2,2,5),np.logspace(-2,2,5)))
    variances = variances = np.concatenate(([0.0],np.logspace(-2,0,3)))
    var_lcs = 1e0
    if criterion == 'rankMax':
        var_refst = 0
        for alpha_reg in alphas:
            placement = CW.Placement(n, n_empty, s, dataset.ds_train, criterion, 
                                     var_lcs, var_refst, 1, alpha_reg)
            placement.placement_allStations()
            placement.save_results(results_path)
    
    elif criterion == 'D_optimal':
        for var_refst in variances:
            placement = CW.Placement(n, n_empty, s, dataset.ds_train, criterion, 
                                     var_lcs, var_refst, 1, 1e0)
            
            placement.placement_allStations()
            placement.save_results(results_path)
    
    elif criterion == 'rankMax_FM':
        var_refst = 0
        for alpha_reg in alphas:
            placement = CW.Placement(n, n_empty, s, dataset.ds_train, criterion, 
                                     var_lcs, var_refst, 1, alpha_reg)
            placement.placement_allStations()
            placement.save_results(results_path)
    
    else:
        raise ValueError(f'Invalid sensor placement algorithm {criterion}')
        

    
def validation_alpha(dataset,n,n_refst,n_empty,s,files_path,criterion='rankMax'):
    """
    Estimate MSE for different alpha values using rankMax solution.
    Using dataset ==  validation data set then select the alpha for the lowest MSE

    Parameters
    ----------
    dataset : pandas dataset
        network measurements
    n : int
        network size
    n_refst : int
        number of reference stations
    n_empty : int
        number of unmonitored locations
    s : int
        signal sparsity level
    files_path : str
        path to rankMax files
    criterion : str, optional
        criteria for solving sensor placement problem. The default is 'rankMax'.

    Returns
    -------
    mse_alphas : dictionary
        MSE for different alpha values
    alpha_min : float
        alpha value for minimum error

    """

    lowrank_basis = LRB.LowRankBasis(dataset.ds_train,s)
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    dataset.project_basis(lowrank_basis.Psi)
        
    alphas = np.concatenate((-np.logspace(-2,2,5),np.logspace(-2,2,5)))
    mse_alphas = {el:np.inf for el in alphas}
    for alpha_reg in alphas:
        print(f'Estimation for alpha: {alpha_reg:.1e}')
        estimation_rankMax = Estimation.Estimation(n, s, n_empty, 
                                                n_refst, 1e0, 0, 
                                                1, alpha_reg,
                                                [], [],[],
                                                lowrank_basis.Psi, '',files_path)
        estimation_rankMax.analytical_estimation(criterion)
        
        mse_alphas[alpha_reg] = estimation_rankMax.mse_analytical_full
        alpha_min = [i for i,v in mse_alphas.items() if v == min(mse_alphas.values())][0]
        print(f'Minimum MSE:\n MSE: {mse_alphas[alpha_min]:.2f}\n alpha: {alpha_min:.1e}')
      
        
    return mse_alphas,alpha_min


        
# =============================================================================
# Estimation
# =============================================================================

def compute_analytical_errors_criteria(dataset,n,n_refst,n_empty,s,alpha_reg,files_path,save_results=False):
    """
    Compute analytical MSE from trace covariance matrix for a given number of reference stations/LCSs in the network and 
    iterated over different variances ratio between refst and LCSs.
    
    Given a number of reference stations (and LCSs) and variances ratio:
        1) Locations are loaded for both criteria
        2) regressor and measurement covariance matrices are computed and the metric measured
        3) The results are stored for both criteria and at different locations.

    Parameters
    ----------
    dataset : DataSet object
        original dataset with measurements
    n : int
        network size
    n_refst : int
        number of RefSt
    n_empty : int
        number of unmonitored locations
    s : int
        signal sparsity level
    alpha_reg : float
        regularization parameter value for rankMax algorithm
    files_path : str
        files directory
    save_results : boolean
        save generated dictionaries

    Returns
    -------
    None.

    """
    #input('Compute analytical RMSE using covariance matrices.\nPress Enter to continue ...')
    # project dataset onto subspace so that data is exactly sparse in a basis
    lowrank_basis = LRB.LowRankBasis(dataset.ds_train,s)
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    dataset.project_basis(lowrank_basis.Psi)
    
    variances = np.concatenate(([0.0],np.logspace(-6,0,7)))
        
    dict_Dopt_var = {el:np.inf for el in variances}
    dict_rankMax_var = {el:np.inf for el in variances}
    dict_rankMaxFM_var = {el:np.inf for el in variances}
    
    for var in variances:
        print(f'Computing estimation for variance: {var:.1e}')
        # estimate using those generated sensors measurements
        
        estimation_Dopt = Estimation.Estimation(n, s, n_empty, 
                                                n_refst, 1e0, var, 
                                                1, alpha_reg,
                                                [], [], [],
                                                lowrank_basis.Psi, files_path,files_path)
        
        estimation_rankMax = Estimation.Estimation(n, s, n_empty, 
                                                n_refst, 1e0, var, 
                                                1, alpha_reg,
                                                [], [], [],
                                                lowrank_basis.Psi, files_path,files_path)
        
        
        
        
       
        
        if var!= 0:
            print('D-optimal estimation')
            estimation_Dopt.analytical_estimation(criterion='D_optimal')
            dict_Dopt_var[var] = [estimation_Dopt.mse_analytical_full,estimation_Dopt.mse_analytical_refst,
                                  estimation_Dopt.mse_analytical_lcs,estimation_Dopt.mse_analytical_unmonitored]
            
            print('RankMax estimation')
            estimation_rankMax.analytical_estimation(criterion='rankMax')
            dict_rankMax_var[var] = [estimation_rankMax.mse_analytical_full,estimation_rankMax.mse_analytical_refst,
                                     estimation_rankMax.mse_analytical_lcs,estimation_rankMax.mse_analytical_unmonitored]
            
           
              
        else:
            print('D-optimal estimation')
            dict_Dopt_var[var] = [np.inf,np.inf,np.inf,np.inf]
            
            print('rankMax estimation')
            estimation_rankMax.analytical_estimation(criterion='rankMax')
            dict_rankMax_var[var] = [estimation_rankMax.mse_analytical_full,estimation_rankMax.mse_analytical_refst,
                                     estimation_rankMax.mse_analytical_lcs,estimation_rankMax.mse_analytical_unmonitored]
           
            
    mse_Dopt = pd.DataFrame(dict_Dopt_var,index=['Full','RefSt','LCS','Unmonitored'],columns=variances).T
    mse_rank = pd.DataFrame(dict_rankMax_var,index=['Full','RefSt','LCS','Unmonitored'],columns=variances).T
     
    
    if save_results:
        fname = f'{results_path}MSE_analytical_Doptimal_{n}N_{n_refst}RefSt_{s}r.pkl'
        with open(fname,'wb') as f:
           pickle.dump(mse_Dopt,f)
        
        fname = f'{results_path}MSE_analytical_rankMax_{n}N_{n_refst}RefSt_{s}r.pkl'
        with open(fname,'wb') as f:
           pickle.dump(mse_rank,f)
           
      
    return mse_Dopt,mse_rank

def local_optimization_swap(dataset,n,n_refst,n_lcs,n_empty,s,var,solution_dist,n_swaps=100):
    """
    Swaps solution found by different criteria in solution_dist and exchange reference station locations with unmonitored ones.
    If an improven on the covariance matrix is found then the new distribution is used.

    Parameters
    ----------
    dataset : dataset type
        measurements data
    n : int
        network size
    n_refst : int
        number of reference stations
    n_lcs : int
        number of LCS
    n_empty : int
        number of unmonitored locations
    s : int
        sparsity
    var: float
        variances ratio
    solution_dist : list
        indices of [LCS,RefSt,Unmonitored] locations
    n_swaps : int, optional
        Maximum number of swap attemps. If reached the algorithm stops. The default is 100.

    Returns
    -------
    list
        New distribution of [LCS,RefSt,Unmonitored] locations.
    float
        New MSE

    """
    
    loc_LCS = solution_dist[0]
    loc_RefSt = solution_dist[1]
    loc_unmonitored = solution_dist[2]
    
    
    # sparse basis
    lowrank_basis = LRB.LowRankBasis(dataset.ds_train,s)
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    dataset.project_basis(lowrank_basis.Psi)
    
    
    # Original covariance matrix
    sensor_placement = SP.SensorPlacement('rankMax', n, s, 
                                          n_refst, n_lcs,
                                          n_empty, 1e0, var)
    sensor_placement.locations = solution_dist
    
    sensor_placement.C_matrix()
    
    if var!=0.0:
        sensor_placement.covariance_matrix_GLS(lowrank_basis.Psi)
    else:
        sensor_placement.covariance_matrix_limit(lowrank_basis.Psi)
    
    #Theta_empty = sensor_placement.C[2]@lowrank_basis.Psi
    #cov_empty = Theta_empty@sensor_placement.Cov@Theta_empty.T
    #mse_orig = np.trace(np.abs(cov_empty))/n_empty
    
    cov_full = lowrank_basis.Psi@sensor_placement.Cov@lowrank_basis.Psi.T
    mse_orig = np.trace(np.abs(cov_full))/n
    
    if len(solution_dist[1]) == 0:
        print('No Reference stations for swapping. Returning original distribution')
        return solution_dist, mse_orig
        
    
    # swap between refst index and unmonitored idx
    count = 0
    new_loc_RefSt = loc_RefSt.copy()
    new_loc_unmonitored = loc_unmonitored.copy()
    
    mse_comparison = mse_orig
    for i in loc_RefSt:
        for j in loc_unmonitored:
            # swap entries
            if j in new_loc_RefSt:
                continue
            
            new_loc_RefSt[np.argwhere(new_loc_RefSt == i)[0][0]] = j
            new_loc_unmonitored[np.argwhere(new_loc_unmonitored==j)[0][0]] = i
            # compute new covariance matrix
            sensor_placement.locations = [loc_LCS,np.sort(new_loc_RefSt),np.sort(new_loc_unmonitored)]
            sensor_placement.C_matrix()
            if var!=0.0:
                sensor_placement.covariance_matrix_GLS(lowrank_basis.Psi)
            else:
                sensor_placement.covariance_matrix_limit(lowrank_basis.Psi)
            
            #Theta_empty = sensor_placement.C[2]@lowrank_basis.Psi
            #cov_empty = Theta_empty@sensor_placement.Cov@Theta_empty.T
            #mse_new = np.trace(np.abs(cov_empty))/n_empty
            
            cov_full = lowrank_basis.Psi@sensor_placement.Cov@lowrank_basis.Psi.T
            mse_new = np.trace(np.abs(cov_full))/n
            
            # check if mse improves
            if mse_new < mse_comparison:
                # skip this entry swapping
                print(f'Improvement when swapping index RefSt {i} with unmonitored {j}\nCurrent {mse_comparison:.2f}\nNew {mse_new:.2f}')
                mse_comparison = mse_new
                break
            else:
                # revert swap
                new_loc_RefSt[np.argwhere(new_loc_RefSt == j)[0][0]] = i
                new_loc_unmonitored[np.argwhere(new_loc_unmonitored==i)[0][0]] = j
            
            count+=1
        if count >n_swaps:
            print(f'Total number of swaps performed: {count}\nTolerance: {n_swaps}.Stopping swaps.')
            break
        
    # Final covariance matrix
    sensor_placement.locations = [loc_LCS,np.sort(new_loc_RefSt),np.sort(new_loc_unmonitored)]
    sensor_placement.C_matrix()
    if var!=0.0:
        sensor_placement.covariance_matrix_GLS(lowrank_basis.Psi)
    else:
        sensor_placement.covariance_matrix_limit(lowrank_basis.Psi)
    
    #Theta_empty = sensor_placement.C[2]@lowrank_basis.Psi
    #cov_empty = Theta_empty@sensor_placement.Cov@Theta_empty.T
    #mse_new = np.trace(np.abs(cov_empty))/n_empty
    cov_full = lowrank_basis.Psi@sensor_placement.Cov@lowrank_basis.Psi.T
    mse_new = np.trace(np.abs(cov_full))/n
    
    
    print(f'Results after {count} swap attemps\nOriginal RefSt distribution: {loc_RefSt}\nNew RefSt distribution: {new_loc_RefSt}\nOriginal unmonitored distribution: {loc_unmonitored}\nNew unmonitored distribution: {new_loc_unmonitored}\nOriginal MSE: {mse_orig}\nNew MSE: {mse_new}')
           
    return [loc_LCS,np.sort(new_loc_RefSt),np.sort(new_loc_unmonitored)], mse_new

            
#%%
if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
    
    #exhaustive_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/Exhaustive_network/'
    
    # network paramteres
    N = 18 #[18,71]
    S = 5 #[5,50 if pollutant == 'O3' else 53]
    POLLUTANT = 'O3' #['O3','NO2']
    
    n_refst_range = np.arange(0,S,1)
    n_empty_range = np.arange(0,N-S+1,1)
    
    #check_network_parameters(N,N_REFST,N_LCS,N_EMPTY,S)
    
    # get reduced dataset
    dataset = load_dataset(files_path,N,POLLUTANT)
    
    # solve convex sensor placement problem
    solve_problem = False
    if solve_problem:
        criterion = 'rankMax' #['rankMax','D_optimal']
        
        print(f'Solving sensor placement problem for network:\n Pollutant: {POLLUTANT}\n N: {N}\n sparsity: {S}\n Algorithm: {criterion}\n Number of unmonitored locations ranging from {n_empty_range[0]} to {n_empty_range[-1]}')
        input('Press Enter to continue...')
        for n_empty in n_empty_range:
            print(f'Solving sensor placement problem for network:\n Pollutant: {POLLUTANT}\n N: {N}\n Unmonitored: {n_empty}\n sparsity: {S}\n Algorithm: {criterion}\n Solving for every number of RefSt from 0 to {N - n_empty}')
            solve_sensor_placement(dataset,N,n_empty,S,results_path,criterion)
        sys.exit()
    
    # determine optimal alpha value for rankMax criterion
    validate = False
    if validate:
        print(f'Validating alpha value for rankMax results.\n Pollutant: {POLLUTANT}\n N: {N}\n sparsity: {S}\n Number of unmonitored locations ranges from {n_empty_range[0]} to {n_empty_range[-1]}\n')
        input('Press enter to continue ...')
        
        df_alphas = pd.DataFrame()
        for n_empty in n_empty_range:
            df = pd.DataFrame(data=None,index=n_refst_range,columns=[n_empty])
            for n_refst in n_refst_range:
                mse_alphas,alpha_min = validation_alpha(dataset,N,n_refst=n_refst,n_empty=n_empty,s=S,
                                              files_path=results_path+'Criteria_comparison/',criterion='rankMax')
                
                df.loc[n_refst,n_empty] = alpha_min
            df_alphas = pd.concat((df_alphas,df),axis=1)
        
        df_alphas.to_csv(results_path+f'Validation_results_{N}N_{S}r.csv')
        sys.exit()
        
    # compute MSE using both criteria and exhaustive configurations forn given variance
    estimate = True
    if estimate:
        var = 1e-2
        df_alphas = pd.read_csv(results_path+f'Criteria_comparison/Validation_results_{N}N_{S}r.csv',index_col=0)
        
        df_rmse_rankMax = pd.DataFrame()
        df_rmse_Dopt = pd.DataFrame()
        
        print(f'Estimating RMSE using both criteria solutions.\n Pollutant: {POLLUTANT}\n N: {N}\n sparsity: {S}\n Variances ratio: {var:.1e}\n Number of unmonitored locations ranges from {n_empty_range[0]} to {n_empty_range[-1]}\n Number of reference stations ranges from 0 to {S-1}\n The rest are LCS up to complete monitored locations')
        input('Press Enter to continue ...')
        for n_empty in n_empty_range:
            df_rankMax = pd.DataFrame(data=None,index = n_refst_range,columns=[n_empty])
            df_Dopt = pd.DataFrame(data=None,index = n_refst_range,columns=[n_empty])
            for n_refst in n_refst_range:
                # load rankMax and Dopt locations
                alpha_reg = df_alphas.loc[n_refst,str(n_empty)]
                fname = results_path+f'Criteria_comparison/DiscreteLocations_rankMax_vs_p0_{N}N_r{S}_pEmpty{n_empty}_alpha{alpha_reg:.1e}.pkl'
                with open(fname,'rb') as f:
                    rankMax_locations = pickle.load(f)
                rankMax_locations = rankMax_locations[n_refst]
        
                fname = results_path+f'Criteria_comparison/DiscreteLocations_D_optimal_vs_p0_{N}N_r{S}_pEmpty{n_empty}_varZero{var:.1e}.pkl'
                with open(fname,'rb') as f:
                    Dopt_locations = pickle.load(f)
                Dopt_locations = Dopt_locations[n_refst]
        
                # local optimization : swapping
                rankMax_locations_swap, rankMax_mse_swap = local_optimization_swap(dataset, N, n_refst, N-n_refst-n_empty, n_empty, 
                                                                                   S, var, rankMax_locations)
                
                rankMax_error_swap = np.sqrt(rankMax_mse_swap)
                
                
                Dopt_locations_swap, Dopt_mse_swap = local_optimization_swap(dataset, N, n_refst, N-n_refst-n_empty, n_empty, 
                                                                                   S, var, Dopt_locations)
                Dopt_error_swap = np.sqrt(Dopt_mse_swap)
                
                df_rankMax.loc[n_refst,n_empty] = rankMax_error_swap
                df_Dopt.loc[n_refst,n_empty] = Dopt_error_swap
            
            df_rmse_rankMax = pd.concat((df_rmse_rankMax,df_rankMax),axis=1)
            df_rmse_Dopt = pd.concat((df_rmse_Dopt,df_Dopt),axis=1)
            
        
        
        df_rmse_rankMax.to_csv(results_path+f'RMSE_rankMax_{N}N_{S}r.csv')
        df_rmse_Dopt.to_csv(results_path+f'RMSE_Dopt_{N}N_{S}r.csv')
        sys.exit()

    
   