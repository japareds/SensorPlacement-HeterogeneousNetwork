#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 12:13:52 2023

@author: jparedes
"""

import itertools
import os
import pandas as pd
import numpy as np
import math
import warnings
import pickle
import sys

import SensorPlacement as SP
import DataSet as DS
import LowRankBasis as LRB
import compute_weights as CW
import Estimation
import Plots


#%% locations algorithms

def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in itertools.islice(it, SIZE)}
        
class ExhaustivePlacement():
    def __init__(self,n,n_refst,n_lcs,n_empty):
        self.n = n
        self.n_refst = n_refst
        self.n_lcs = n_lcs
        self.n_empty = n_empty
        
        print(f'Exhaustive placement network:\n{n} possible locations\n{n_refst} Ref.St.\n{n_lcs} LCS\n{n_empty} unmonitored locations')
        if n_refst + n_lcs + n_empty != n:
            warnings.warn('Total sum of sensors and unmonitored locations mismatch total number of possible locations.')
        if any(i <0 for i in [n,n_refst,n_lcs,n_empty]):
            warnings.warn('There is a negative number')
        
        self.num_distributions = math.comb(self.n,self.n_refst)*math.comb(self.n-self.n_refst,self.n_lcs)
  
    def distribute(self):
        """
        Generate all locations where n_refst reference stations and n_lcs LCSs could be in a network of size n

        Returns
        -------
        None.

        """
        print(f'Generating all possible combinations for distributing {self.n_refst + self.n_lcs} sensors in total on a network of {self.n} locations.\n{self.n_refst} reference stations\n{self.n_lcs} LCS\n{self.n_empty} unmonitored locations')
        
        print(f'In total there are {math.comb(self.n,self.n_refst)*math.comb(self.n-self.n_refst,self.n_lcs):.2e} combinations')
        input('Press Enter to continue ...')
        all_locations = np.arange(self.n)
        # all possible locations for reference stations
        self.loc_refst = {el:np.array(i) for el,i in zip(range(math.comb(self.n,self.n_refst)),itertools.combinations(np.arange(self.n), self.n_refst))}
        
        self.loc_lcs = {el:[] for el in range(math.comb(self.n,self.n_refst))}
        self.loc_empty = {el:[] for el in range(math.comb(self.n,self.n_refst))}
        
        # given certain reference stations distribution: get possible locations for LCSs
        for idx_refst in self.loc_refst:
            possible_lcs_locations = np.setdiff1d(all_locations, self.loc_refst[idx_refst])
            self.loc_lcs[idx_refst] = [np.array(i) for i in itertools.combinations(possible_lcs_locations,self.n_lcs)]
              
        # place unmonitored locations
        for idx_refst in self.loc_refst:
            for idx_lcs in range(len(self.loc_lcs[idx_refst])):
                occupied_locations = np.concatenate((self.loc_refst[idx_refst],self.loc_lcs[idx_refst][idx_lcs]))
                self.loc_empty[idx_refst].append(np.setdiff1d(all_locations, occupied_locations))
                
        
        print(f'{len(self.loc_refst)} possible distributions of reference stations\nFor each of those combinations there are {len(self.loc_lcs[0])} LCSs configurations\n')
        
    def sort_locations(self):
        """
        Rearrange locations index as [idx_lcs,idx_refst,idx_empty]

        Returns
        -------
        None.

        """
        self.locations = {el:0 for el in range(math.comb(self.n,self.n_refst)*math.comb(self.n-self.n_refst,self.n_lcs))}
        idx = 0
        for idx_refst in self.loc_refst:
            for idx_lcs in range(math.comb(self.n - self.n_refst , self.n_lcs)):
                self.locations[idx] = [self.loc_lcs[idx_refst][idx_lcs],
                                       self.loc_refst[idx_refst],
                                       self.loc_empty[idx_refst][idx_lcs]]
                idx+=1
                
    
    def save_locations(self,path):
        """
        Save dictionaries of locations generated

        Parameters
        ----------
        path : str
            path to files

        Returns
        -------
        None.

        """
        fname = path+f'exhaustiveLocations_RefSt_{self.n}total_{self.n_refst}refSt_{self.n_lcs}LCS_{self.n_empty}empty.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.loc_refst,f)
        
        fname = path+f'exhaustiveLocations_LCS_{self.n}total_{self.n_refst}refSt_{self.n_lcs}LCS_{self.n_empty}empty.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.loc_lcs,f)
        
        fname = path+f'exhaustiveLocations_Empty_{self.n}total_{self.n_refst}refSt_{self.n_lcs}LCS_{self.n_empty}empty.pkl'
        with open(fname,'wb') as f:
            pickle.dump(self.loc_empty,f)
    
    def load_locations(self,path):
        """
        Load locations for given network paramters

        Parameters
        ----------
        path : str
            path to files

        Returns
        -------
        None.

        """
        fname = path+f'exhaustiveLocations_RefSt_{self.n}total_{self.n_refst}refSt_{self.n_lcs}LCS_{self.n_empty}empty.pkl'
        with open(fname,'rb') as f:
            self.loc_refst = pickle.load(f)
        
        fname = path+f'exhaustiveLocations_LCS_{self.n}total_{self.n_refst}refSt_{self.n_lcs}LCS_{self.n_empty}empty.pkl'
        with open(fname,'rb') as f:
            self.loc_lcs = pickle.load(f)
        
        fname = path+f'exhaustiveLocations_Empty_{self.n}total_{self.n_refst}refSt_{self.n_lcs}LCS_{self.n_empty}empty.pkl'
        with open(fname,'rb') as f:
            self.loc_empty = pickle.load(f)
            
    def load_errors_exhaustive(self,loc,var,n_refst,s,path):
        """
        Load MSE for all possible configurations for a network with n_refst RefSt and variances ratio var

        Parameters
        ----------
        loc : str
            locations of the network
        var : float
            variances ratio
        path : str
            File's directory

        Returns
        -------
        None.

        """
        
        fname = path+f'MSE_analytical_exhaustive_{loc}_{n_refst}RefSt_var{var:.1e}_r{s}.pkl'
        with open(fname,'rb') as f:
            self.exhaustive_mse = pickle.load(f)
        # sort values and their repective index for variance == 0.0
        
        try:
            self.errors_sorted = np.sqrt(np.sort([i[0] for i in self.exhaustive_mse.values()]))
            self.loc_sorted = np.argsort([i[0] for i in self.exhaustive_mse.values()])
            
        except:
            self.errors_sorted = np.sqrt(np.sort([i for i in self.exhaustive_mse.values()]))
            self.loc_sorted = np.argsort([i for i in self.exhaustive_mse.values()])
           
    def find_specific_configuration(self,config):
        """
        Return index at which distribution given in config is located among all the possible distributions

        Parameters
        ----------
        config : list
            Sensor distributions in the network. Entry 0 has LCSs locations. Entry 1 has RefSt locations and Entry 2 has unmonitored locations

        Returns
        -------
        idx : int
            index in the whole possible configurations

        """
        
        idx = np.inf
        if len(config[0]) == 0:
            return idx
        
        for i in range(self.num_distributions):
            loc_LCS, loc_RefSt, loc_unmonitored = self.locations[i]
            try:
                if (config[0] == loc_LCS).all() and (config[1] == loc_RefSt).all() and (config[2] == loc_unmonitored).all():
                    idx = i
                    print(f'Equivalence at index {idx}')
                    return idx
            except:
                pass
        return idx
            
        
           
        
#%%
def get_exhaustive_distribution(exhaustive_placement,exhaustive_path):
    try:
        exhaustive_placement.load_locations(exhaustive_path)
    except:
        exhaustive_placement.distribute()
        exhaustive_placement.save_locations(results_path)
    
def load_dataset():
    pollutant = 'O3' #['O3','NO2']
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

def solve_sensor_placement(dataset,criterion='rankMax'):
    alphas = np.concatenate((-np.logspace(-2,2,5),np.logspace(-2,2,5)))
    variances = variances = np.concatenate(([0.0],np.logspace(-6,0,7)))
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
            
def solution_swap(dataset,s,var,solution_dist,n_swaps=100):
    """
    Swaps solution found by different criteria in solution_dist and exchange reference station locations with unmonitored ones.
    If an improven on the covariance matrix is found then the new distribution is used.

    Parameters
    ----------
    dataset : dataset type
        measurements data
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
    
    if len(loc_unmonitored) == 0:
        print('No solution for criterion.')
        return [[],[],[]],np.inf
    
    # sparse basis
    lowrank_basis = LRB.LowRankBasis(dataset.ds_train,s)
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    dataset.project_basis(lowrank_basis.Psi)
    
    
    # covariance matrix
    sensor_placement = SP.SensorPlacement('rankMax', n, s, 
                                          n_refst, n_lcs,
                                          n_empty, 1e0, var)
    sensor_placement.locations = solution_dist
    
    sensor_placement.C_matrix()
    
    if var!=0.0:
        sensor_placement.covariance_matrix_GLS(lowrank_basis.Psi)
    else:
        sensor_placement.covariance_matrix_limit(lowrank_basis.Psi)
    
    Theta_empty = sensor_placement.C[2]@lowrank_basis.Psi
    cov_empty = Theta_empty@sensor_placement.Cov@Theta_empty.T
    mse_orig = np.trace(np.abs(cov_empty))/n_empty
    
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
            
            Theta_empty = sensor_placement.C[2]@lowrank_basis.Psi
            cov_empty = Theta_empty@sensor_placement.Cov@Theta_empty.T
            mse_new = np.trace(np.abs(cov_empty))/n_empty
            
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
    
    Theta_empty = sensor_placement.C[2]@lowrank_basis.Psi
    cov_empty = Theta_empty@sensor_placement.Cov@Theta_empty.T
    mse_new = np.trace(np.abs(cov_empty))/n_empty
    print(f'Results after {count} swap attemps\nOriginal RefSt distribution: {loc_RefSt}\nNew RefSt distribution: {new_loc_RefSt}\nOriginal unmonitored distribution: {loc_unmonitored}\nNew unmonitored distribution: {new_loc_unmonitored}\nOriginal MSE: {mse_orig}\nNew MSE: {mse_new}')
           
    return [loc_LCS,np.sort(new_loc_RefSt),np.sort(new_loc_unmonitored)], mse_new
            
            
    
    
def validation_alpha(dataset,n_refst,s,criterion='rankMax'):
    """
    Estimate MSE for different alpha values using rankMax solution.
    Using dataset ==  validation data set then select the alpha for the lowest MSE

    Parameters
    ----------
    dataset : pandas dataset
        network measurements
    n_refst : int
        number of reference stations
    criterion : str, optional
        criteria for solving sensor placement problem. The default is 'rankMax'.

    Returns
    -------
    mse_alphas : dictionary
        MSE for different alpha values

    """

    lowrank_basis = LRB.LowRankBasis(dataset.ds_train,s)
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    dataset.project_basis(lowrank_basis.Psi)
        
    alphas = np.concatenate((-np.logspace(-2,2,5),np.logspace(-2,2,5)))
    mse_alphas = {el:np.inf for el in alphas}
    for alpha_reg in alphas:
        estimation_rankMax = Estimation.Estimation(n, s, n_empty, 
                                                n_refst, 1e0, 0, 
                                                1, alpha_reg,
                                                [], [],[],
                                                lowrank_basis.Psi, '',exhaustive_path)
        estimation_rankMax.analytical_estimation(criterion)
        
        mse_alphas[alpha_reg] = estimation_rankMax.mse_analytical_full
        
    return mse_alphas

def compute_analytical_errors_criteria(dataset,n_refst,alpha_reg,save_results=False):
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
    save_results : boolean
        save generated dictionaries

    Returns
    -------
    None.

    """
    input('Compute analytical RMSE using covariance matrices.\nPress Enter to continue ...')
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
        # estimate using those generated sensors measurements
        
        estimation_Dopt = Estimation.Estimation(n, s, n_empty, 
                                                n_refst, 1e0, var, 
                                                1, alpha_reg,
                                                [], [], [],
                                                lowrank_basis.Psi, exhaustive_path,exhaustive_path)
        
        estimation_rankMax = Estimation.Estimation(n, s, n_empty, 
                                                n_refst, 1e0, var, 
                                                1, alpha_reg,
                                                [], [], [],
                                                lowrank_basis.Psi, exhaustive_path,exhaustive_path)
        
        estimation_rankMaxFM = Estimation.Estimation(n, s, n_empty, 
                                                n_refst, 1e0, var, 
                                                1, alpha_reg,
                                                [], [], [],
                                                lowrank_basis.Psi, exhaustive_path,exhaustive_path)
        
        
       
        
        if var!= 0:
        
            estimation_Dopt.analytical_estimation(criterion='D_optimal')
            dict_Dopt_var[var] = [estimation_Dopt.mse_analytical_full,estimation_Dopt.mse_analytical_refst,
                                  estimation_Dopt.mse_analytical_lcs,estimation_Dopt.mse_analytical_unmonitored]
            
            estimation_rankMax.analytical_estimation(criterion='rankMax')
            dict_rankMax_var[var] = [estimation_rankMax.mse_analytical_full,estimation_rankMax.mse_analytical_refst,
                                     estimation_rankMax.mse_analytical_lcs,estimation_rankMax.mse_analytical_unmonitored]
            
            estimation_rankMaxFM.analytical_estimation(criterion='rankMax_FM')
            dict_rankMaxFM_var[var] = [estimation_rankMaxFM.mse_analytical_full,estimation_rankMaxFM.mse_analytical_refst,
                                     estimation_rankMaxFM.mse_analytical_lcs,estimation_rankMaxFM.mse_analytical_unmonitored]
              
        else:
            dict_Dopt_var[var] = [np.inf,np.inf,np.inf,np.inf]
            
            estimation_rankMax.analytical_estimation(criterion='rankMax')
            dict_rankMax_var[var] = [estimation_rankMax.mse_analytical_full,estimation_rankMax.mse_analytical_refst,
                                     estimation_rankMax.mse_analytical_lcs,estimation_rankMax.mse_analytical_unmonitored]
            estimation_rankMaxFM.analytical_estimation(criterion='rankMax_FM')
            dict_rankMaxFM_var[var] = [estimation_rankMaxFM.mse_analytical_full,estimation_rankMaxFM.mse_analytical_refst,
                                     estimation_rankMaxFM.mse_analytical_lcs,estimation_rankMaxFM.mse_analytical_unmonitored]
            
    mse_Dopt = pd.DataFrame(dict_Dopt_var,index=['Full','RefSt','LCS','Unmonitored'],columns=variances).T
    mse_rank = pd.DataFrame(dict_rankMax_var,index=['Full','RefSt','LCS','Unmonitored'],columns=variances).T
    mse_rankFM = pd.DataFrame(dict_rankMaxFM_var,index=['Full','RefSt','LCS','Unmonitored'],columns=variances).T
    
    
    if save_results:
        fname = f'{results_path}MSE_analytical_Doptimal_{n}nTot_{n_refst}RefSt_r{s}.pkl'
        with open(fname,'wb') as f:
           pickle.dump(mse_Dopt,f)
        
        fname = f'{results_path}MSE_analytical_rankMax_{n}nTot_{n_refst}RefSt_r{s}.pkl'
        with open(fname,'wb') as f:
           pickle.dump(mse_rank,f)
           
        fname = f'{results_path}MSE_analytical_rankMaxFM_{n}nTot_{n_refst}RefSt_r{s}.pkl'
        with open(fname,'wb') as f:
           pickle.dump(mse_rankFM,f)
           
           
      
    return mse_Dopt,mse_rank,mse_rankFM
        

def compute_analytical_errors_exhaustive(dataset,n_refst,var,num_el_chunk = 5000,save_results=False):
    """
    Compute analytical MSE from trace covariance matrix for multiple number of reference stations in the network and 
    different variances ratio between refst and LCSs.
    
    The computation is heavily memory expensive.
    
  
    Parameters
    ----------
    dataset : DataSet object
        original dataset with measurements
    n_refst : int
        number of reference stations in the network
    var : float
        variances ratio
    save_results : boolean
        save generated dictionaries

    Returns
    -------
    None.

    """
    print('Compute analytical RMSE using covariance matrices.\nComputing ALL possible locations!')
    #input('Press Enter to continue ...')
    # project dataset onto subspace so that data is exactly sparse in a basis
    lowrank_basis = LRB.LowRankBasis(dataset.ds_train,s)
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    dataset.project_basis(lowrank_basis.Psi)
    
    it = 0
    
    for item in chunks(exhaustive_placement.locations, num_el_chunk): # split dict into chunk for memory
        new_locs = {el:i for el,i in zip(range(num_el_chunk),item.values())}    
        estimation_exhaustive = Estimation.Estimation(n, s, 
                                                      n_empty, n_refst, 
                                                      1e0, var, 
                                                      1, 1e0,
                                                      [], [], [],
                                                      lowrank_basis.Psi, exhaustive_path,exhaustive_path)
        estimation_exhaustive.analytical_estimation_exhaustive(new_locs)
        it+=1
    
        if save_results:
            fname = f'{results_path}MSE_analytical_exhaustive_Full_{n_refst}RefSt_var{var:.1e}_r{s}_it{it}.pkl'
            with open(fname,'wb') as f:
               pickle.dump(estimation_exhaustive.mse_analytical_full,f)
               
            fname = f'{results_path}MSE_analytical_exhaustive_Unmonitored_{n_refst}RefSt_var{var:.1e}_r{s}_it{it}.pkl'
            with open(fname,'wb') as f:
               pickle.dump(estimation_exhaustive.mse_analytical_unmonitored,f)
            
            fname = f'{results_path}MSE_analytical_exhaustive_RefSt_{n_refst}RefSt_var{var:.1e}_r{s}_it{it}.pkl'
            with open(fname,'wb') as f:
               pickle.dump(estimation_exhaustive.mse_analytical_refst,f)
            
            fname = f'{results_path}MSE_analytical_exhaustive_LCS_{n_refst}RefSt_var{var:.1e}_r{s}_it{it}.pkl'
            with open(fname,'wb') as f:
               pickle.dump(estimation_exhaustive.mse_analytical_lcs,f)
        
    return estimation_exhaustive

def join_exhaustive_placement_iterations(path_files,var,n_it,locations='Unmonitored'):
    
    values = np.array([])
    for it in np.arange(1,n_it+1,1):
        fname = path_files+f'MSE_analytical_exhaustive_{locations}_{n_refst}RefSt_var{var:.1e}_r{s}_it{it}.pkl'
        with open(fname,'rb') as f:
            unmonitored_mse_it = pickle.load(f)
        values = np.append(values,[i[0] for i in unmonitored_mse_it.values()])
    dict_values = {i:values[i] for i in range(len(values))}
    fname = path_files+f'MSE_analytical_exhaustive_{locations}_{n_refst}RefSt_var{var:.1e}_r{s}.pkl'
    with open(fname,'wb') as f:
       pickle.dump(dict_values,f)
    

def load_error_results_criteria(loc,var,s,n_refst,alpha_reg,path):
    """
    Load errors obtained by different placement criteria and their respective network configurations
    for a certain number of refst/LCSs in the network and certain variances ratio.
    
    """
    
    # load criteria results
    
    ## rankMax results
    fname = path+f'MSE_analytical_rankMax_{n}nTot_{n_refst}RefSt_r{s}.pkl'
    with open(fname,'rb') as f:
        rankMax_mse = pickle.load(f)
    rankMax_error = np.sqrt(rankMax_mse.loc[var][loc])
    
    fname = path+f'DiscreteLocations_rankMax_vs_p0_r{s}_pEmpty{n_empty}_alpha{alpha_reg:.1e}.pkl'
    with open(fname,'rb') as f:
        rankMax_locations = pickle.load(f)
    rankMax_locations = rankMax_locations[n_refst]
    
    ## Hybrid rankMax - Dopt
    fname = path+f'MSE_analytical_rankMaxFM_{n}nTot_{n_refst}RefSt_r{s}.pkl'
    with open(fname,'rb') as f:
        rankMaxFM_mse = pickle.load(f)
    rankMaxFM_error = np.sqrt(rankMaxFM_mse.loc[var][loc])
    
    fname = path+f'DiscreteLocations_rankMax_FM_vs_p0_r{s}_pEmpty{n_empty}_alpha{alpha_reg:.1e}.pkl'
    with open(fname,'rb') as f:
        rankMaxFM_locations = pickle.load(f)
    rankMaxFM_locations = rankMaxFM_locations[n_refst]
        
    ## Doptimal
    fname = path+f'MSE_analytical_Doptimal_{n}nTot_{n_refst}RefSt_r{s}.pkl'
    with open(fname,'rb') as f:
        Dopt_mse = pickle.load(f)
    Dopt_error = np.sqrt(Dopt_mse.loc[var][loc])
    
    if Dopt_error == np.inf:
        Dopt_locations = np.array([[],[],[]])
    else:
        fname = path+f'DiscreteLocations_D_optimal_vs_p0_r{s}_pEmpty{n_empty}_varZero{var:.1e}.pkl'
        with open(fname,'rb') as f:
            Dopt_locations = pickle.load(f)
        Dopt_locations = Dopt_locations[n_refst]
        
    
    
    
    return Dopt_error,rankMax_error,rankMaxFM_error, Dopt_locations, rankMax_locations, rankMaxFM_locations
       
     
    
    

def figure_exhaustive_criteria(errors_sorted,locations_sorted,rank_error,rankFM_error,Dopt_error,idx_rank,idx_rankFM,idx_Dopt,n_refst,var,save_fig=False):
    plots = Plots.Plots(save_path=results_path,marker_size=1,
                        fs_label=7,fs_ticks=7,fs_legend=5,fs_title=10,
                        show_plots=True)
    
    plots.plot_ranking_error(errors_sorted, locations_sorted,
                             rank_error,rankFM_error, Dopt_error, 
                             idx_rank,idx_rankFM,idx_Dopt,
                             n_refst,var,s,save_fig)
   
   


        
#%%
if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    exhaustive_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/Exhaustive_network/'
    
    # network paramteres
    n = 18
    n_refst = 5
    n_lcs = 10
    n_empty = 3
    s = 15#n_refst + n_lcs
    
    # load or generate exhaustive sensors distribution
    exhaustive_placement = ExhaustivePlacement(n, n_refst, n_lcs, n_empty)
    get_exhaustive_distribution(exhaustive_placement, exhaustive_path)
    exhaustive_placement.sort_locations()
    
    # get reduced dataset
    dataset = load_dataset()
    sys.exit()
    
    # solve convex sensor placement problem
    solve = False
    if solve:
        solve_sensor_placement(dataset,criterion='D_optimal')
        sys.exit()
    # validate alpha rankMax
    validate = False
    if validate:
        mse_alphas = validation_alpha(dataset, n_refst,s,criterion='rankMax')
        sys.exit()
    
    # compute MSE using both criteria and exhaustive configurations forn given variance
    estimate = False
    if estimate:
        if n_refst == 5 and s==15:
            alpha_reg = 1e0
        elif n_refst==10 and s==15:
            alpha_reg = -1e1
        elif n_refst == 7 and s==15:
            alpha_reg = 1e-1
        
        elif n_refst == 5 and s==7:
            alpha_reg = -1e1
        elif n_refst == 7 and s==7:
            alpha_reg = 1e2
        elif n_refst == 10 and s == 7:
            alpha_reg = 1e2
            
            
            
        mse_Dopt,mse_rank,mse_rankFM = compute_analytical_errors_criteria(dataset,n_refst,alpha_reg,save_results=True)
        
        variances = np.concatenate(([0.0],np.logspace(-6,0,7)))
        num_el_chunk = 5000
        num_files = int(np.ceil(exhaustive_placement.num_distributions/num_el_chunk))
        for var in variances:
        # exhaustive estimation: splitted into chunks
            estimation_exhaustive = compute_analytical_errors_exhaustive(dataset, n_refst, var,num_el_chunk,save_results=True)
            
        
            join_exhaustive_placement_iterations(results_path,var,num_files,locations='Unmonitored')
            join_exhaustive_placement_iterations(results_path,var,num_files,locations='Full')
            join_exhaustive_placement_iterations(results_path,var,num_files,locations='RefSt')
            join_exhaustive_placement_iterations(results_path,var,num_files,locations='LCS')
        
     
    #%%
    # plot comparison criteria location in the exhaustive ranks
    show_figures = True
    if show_figures:
        
        ## Plot RMSE vs ranking for given variance and sensors proportions. Highlight criteria locations in the ranking.
        
        loc = 'Unmonitored' # ['Full','RefSt','LCS','Unmonitored']
        var = 1e0
        if n_refst == 5:
            alpha_reg = 1e0
        elif n_refst == 10:
            alpha_reg = -1e1
        elif n_refst == 7:
            alpha_reg = 1e-1
        elif n_refst == 5 and s==7:
            alpha_reg = -1e1
        elif n_refst == 10 and s == 7:
            alpha_reg = 1e2
              
        
        
        
        #load criteria results
        Dopt_error, rankMax_error,rankMaxFM_error, Dopt_locations, rankMax_locations, rankMaxFM_locations = load_error_results_criteria(loc, var,
                                                                                                                                        s,n_refst,
                                                                                                                                        alpha_reg,
                                                                                                                                        exhaustive_path)
        
        # improve results via swapping
        
        rankMax_locations_swap, rankMax_mse_swap = solution_swap(dataset, s, var,rankMax_locations)
        rankMax_error_swap = np.sqrt(rankMax_mse_swap)
        rankMaxFM_locations_swap, rankMaxFM_mse_swap = solution_swap(dataset, s,var, rankMaxFM_locations)
        rankMaxFM_error_swap = np.sqrt(rankMaxFM_mse_swap)
        Dopt_locations_swap, Dopt_mse_swap = solution_swap(dataset, s,var, Dopt_locations)
        Dopt_error_swap = np.sqrt(Dopt_mse_swap)
        
        
        
        # load exhaustive results
        exhaustive_placement.load_errors_exhaustive(loc, var, n_refst,s,exhaustive_path)
        
        print(f'Errors obtained for variances ratio {var:.1e}\n Global minimum: {exhaustive_placement.errors_sorted.min()}\n\nDifferent criteria results\n Doptimal: {Dopt_error}\n Doptimal-swap: {Dopt_error_swap}\n\n rankMax: {rankMax_error}\n rankMax_swap: {rankMax_error_swap}\n\n Hybrid rankMax-FM: {rankMaxFM_error}\n Hybrid rankMax-FM-swap: {rankMaxFM_error_swap}')

        #index of criteria in exhaustive
        idx_rankMax = exhaustive_placement.find_specific_configuration(rankMax_locations_swap)
        idx_rankMaxFM = exhaustive_placement.find_specific_configuration(rankMaxFM_locations_swap)
        idx_Dopt = exhaustive_placement.find_specific_configuration(Dopt_locations_swap)
        
        
        figure_exhaustive_criteria(exhaustive_placement.errors_sorted,exhaustive_placement.loc_sorted,
                                   rankMax_error,rankMaxFM_error,Dopt_error,
                                   idx_rankMax,idx_rankMaxFM,idx_Dopt,
                                   n_refst,var,save_fig=True)
        #%%
        ## plot RMSE vs ranking at var==0. Highlight criteria location in the ranking. REPEAT RMSE vs ranking at var==0 for results with var!=0
        var = 0.0
        loc = 'Unmonitored' # ['Full','RefSt','LCS','Unmonitored']
        if n_refst == 5 and s==15:
            alpha_reg = 1e0
        elif n_refst == 10 and s==15:
            alpha_reg = -1e1
        elif n_refst == 7 and s==15:
            alpha_reg = 1e-1
            
        elif n_refst == 5 and s==7:
            alpha_reg = -1e1
        
        
        Dopt_error, rankMax_error,rankMaxFM_error, Dopt_locations, rankMax_locations, rankMaxFM_locations = load_error_results_criteria(loc, var,
                                                                                                                                        s,n_refst,
                                                                                                                                        alpha_reg,
                                                                                                                                        exhaustive_path)
        # improve results via swapping
        
        rankMax_locations_swap, rankMax_mse_swap = solution_swap(dataset, s, var,rankMax_locations)
        rankMax_error_swap = np.sqrt(rankMax_mse_swap)
        rankMaxFM_locations_swap, rankMaxFM_mse_swap = solution_swap(dataset, s,var, rankMaxFM_locations)
        rankMaxFM_error_swap = np.sqrt(rankMaxFM_mse_swap)
        Dopt_locations_swap, Dopt_mse_swap = solution_swap(dataset, s,var, Dopt_locations)
        Dopt_error_swap = np.sqrt(Dopt_mse_swap)
        # load exhaustive results for var==0
        exhaustive_placement.load_errors_exhaustive(loc, var, n_refst,s,exhaustive_path)
        errors_sorted_zero= exhaustive_placement.errors_sorted.copy()
        locations_sorted_zero = exhaustive_placement.loc_sorted.copy()
        
        #index of criteria in exhaustive
        idx_rankMax = exhaustive_placement.find_specific_configuration(rankMax_locations_swap)
        idx_rankMaxFM = exhaustive_placement.find_specific_configuration(rankMaxFM_locations_swap)
        idx_Dopt = exhaustive_placement.find_specific_configuration(Dopt_locations_swap)
        
        # load exhaustive results for var!=0 and sort them according to index for var==0
        variances = np.logspace(-2,0,2)
        errors_sorted = {el:[] for el in variances}
        for v in variances:
            exhaustive_placement.load_errors_exhaustive(loc, v, n_refst,s,exhaustive_path)
            errors_sorted[v] = np.sqrt(np.array([i for i in exhaustive_placement.exhaustive_mse.values()]))[locations_sorted_zero]
        
        # Dopt has no solution for var==0. Recompute for smallest var!=0
        if idx_Dopt == np.inf:
            v = np.min(variances)
            Dopt_error,_,_, Dopt_locations, _, _ = load_error_results_criteria(loc, v,
                                                                               s,n_refst,
                                                                               alpha_reg,
                                                                               exhaustive_path)
            Dopt_locations_swap, Dopt_mse_swap = solution_swap(dataset, s,v, Dopt_locations)
            Dopt_error_swap = np.sqrt(Dopt_mse_swap)
            
            exhaustive_placement.load_errors_exhaustive(loc, var, n_refst,s,exhaustive_path)
            idx_Dopt = exhaustive_placement.find_specific_configuration(Dopt_locations_swap)
            
            
        plots = Plots.Plots(save_path=results_path,marker_size=1,
                            fs_label=7,fs_ticks=7,fs_legend=6,fs_title=10,
                            show_plots=True)
        plots.ranking_error_comparison(errors_sorted_zero, locations_sorted_zero, 
                                       Dopt_error, errors_sorted,
                                       variances, idx_rankMax, idx_rankMaxFM, idx_Dopt,
                                       n_refst,s,save_fig=True)
        
        
        
        
       
    
    
    