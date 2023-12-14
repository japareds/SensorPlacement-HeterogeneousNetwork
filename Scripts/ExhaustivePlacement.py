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
import matplotlib as mpl
import matplotlib.pyplot as plt

import SensorPlacement as SP
import DataSet as DS
import LowRankBasis as LRB
import compute_weights as CW
import Estimation



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


def compute_analytical_errors_criteria(dataset,n_refst,n_empty,s,alpha_reg,save_results=False):
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
    n_refst : int
        number of RefSt
    n_empty : int
        number of unmonitored locations
    s : int
        signal sparsity level
    alpha_reg : float
        regularization parameter value for rankMax algorithm
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
    #dict_rankMaxFM_var = {el:np.inf for el in variances}
    
    for var in variances:
        print(f'Computing estimation for variance: {var:.1e}')
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
        
        # estimation_rankMaxFM = Estimation.Estimation(n, s, n_empty, 
        #                                         n_refst, 1e0, var, 
        #                                         1, alpha_reg,
        #                                         [], [], [],
        #                                         lowrank_basis.Psi, exhaustive_path,exhaustive_path)
        
        
       
        
        if var!= 0:
            print('D-optimal estimation')
            estimation_Dopt.analytical_estimation(criterion='D_optimal')
            dict_Dopt_var[var] = [estimation_Dopt.mse_analytical_full,estimation_Dopt.mse_analytical_refst,
                                  estimation_Dopt.mse_analytical_lcs,estimation_Dopt.mse_analytical_unmonitored]
            
            print('RankMax estimation')
            estimation_rankMax.analytical_estimation(criterion='rankMax')
            dict_rankMax_var[var] = [estimation_rankMax.mse_analytical_full,estimation_rankMax.mse_analytical_refst,
                                     estimation_rankMax.mse_analytical_lcs,estimation_rankMax.mse_analytical_unmonitored]
            
            # estimation_rankMaxFM.analytical_estimation(criterion='rankMax_FM')
            # dict_rankMaxFM_var[var] = [estimation_rankMaxFM.mse_analytical_full,estimation_rankMaxFM.mse_analytical_refst,
            #                          estimation_rankMaxFM.mse_analytical_lcs,estimation_rankMaxFM.mse_analytical_unmonitored]
              
        else:
            print('D-optimal estimation')
            dict_Dopt_var[var] = [np.inf,np.inf,np.inf,np.inf]
            
            print('rankMax estimation')
            estimation_rankMax.analytical_estimation(criterion='rankMax')
            dict_rankMax_var[var] = [estimation_rankMax.mse_analytical_full,estimation_rankMax.mse_analytical_refst,
                                     estimation_rankMax.mse_analytical_lcs,estimation_rankMax.mse_analytical_unmonitored]
            # estimation_rankMaxFM.analytical_estimation(criterion='rankMax_FM')
            # dict_rankMaxFM_var[var] = [estimation_rankMaxFM.mse_analytical_full,estimation_rankMaxFM.mse_analytical_refst,
            #                          estimation_rankMaxFM.mse_analytical_lcs,estimation_rankMaxFM.mse_analytical_unmonitored]
            
    mse_Dopt = pd.DataFrame(dict_Dopt_var,index=['Full','RefSt','LCS','Unmonitored'],columns=variances).T
    mse_rank = pd.DataFrame(dict_rankMax_var,index=['Full','RefSt','LCS','Unmonitored'],columns=variances).T
    # mse_rankFM = pd.DataFrame(dict_rankMaxFM_var,index=['Full','RefSt','LCS','Unmonitored'],columns=variances).T
    
    
    if save_results:
        fname = f'{results_path}MSE_analytical_Doptimal_{n}nTot_{n_refst}RefSt_r{s}.pkl'
        with open(fname,'wb') as f:
           pickle.dump(mse_Dopt,f)
        
        fname = f'{results_path}MSE_analytical_rankMax_{n}nTot_{n_refst}RefSt_r{s}.pkl'
        with open(fname,'wb') as f:
           pickle.dump(mse_rank,f)
           
        # fname = f'{results_path}MSE_analytical_rankMaxFM_{n}nTot_{n_refst}RefSt_r{s}.pkl'
        # with open(fname,'wb') as f:
        #    pickle.dump(mse_rankFM,f)
           
           
      
    return mse_Dopt,mse_rank


def compute_analytical_errors_all_proportions(dataset,n,n_empty,s):
    """
    Computes MSE for all possible number of reference stations in the network of size n and sparsity s.
    The station has n_empty unmonitored locations.
    
    The number of ref.st. goes from 0 to s-1
    The rest of sensors are LCSs

    Parameters
    ----------
    dataset : dataset object
        dataset with measurements
    n : int
        network size
    n_empty : int
        number of unmonitored sites in the network
    s : int
        sparsity level

    Returns
    -------
    mse_Dopt : dict
        MSE for each number of refst in the network using Doptimal solutions
    mse_rank : dict
        MSE for each number of refst in the network using rankMax solutions

    """
    if s == 15 and n_empty == 3:
        alphas = {0:1e0,1:-1e-2,2:-1e-2,
                  3:-1e-2,4:1e0,5:1e0,
                  6:1e0,7:1e-1,8:1e-2,
                  9:-1e2,10:-1e1,11:-1e1,
                  12:1e-2,13:1e-2,14:1e-2}
    
    elif s==15 and n_empty ==2:
        alphas = {0:-1e-2,1:1e2,2:1e-2,
                  3:1e-2,4:1e0,5:1e0,
                  6:1e0,7:1e0,8:1e0,
                  9:1e-2,10:-1e-2,11:-1e-1,
                  12:-1e2,13:-1e-1,14:-1e-2}
        
    elif s==15 and n_empty == 1:
        alphas = {0:-1e-2,1:1e-2,2:-1e-2,
                  3:-1e-2,4:-1e1,5:1e1,
                  6:-1e1,7:-1e1,8:-1e1,
                  9:-1e2,10:-1e1,11:-1e1,
                  12:-1e2,13:-1e1,14:-1e1}
        
    elif s==15 and n_empty == 0:
        alphas = {0:-1e-2,1:-1e-2,2:-1e-2,
                  3:-1e1,4:-1e1,5:-1e-2,
                  6:-1e1,7:-1e1,8:-1e1,
                  9:-1e1,10:-1e1,11:-1e1,
                  12:-1e2,13:-1e1,14:-1e1}
        
        
    total_refst = [i for i in alphas.keys()]
    mse_Dopt = {el:np.inf for el in total_refst}
    mse_rank = {el:np.inf for el in total_refst}
    
    
    for num_refst in total_refst:
        print(f'computing for {num_refst} Reference stations')
        result_Dopt,result_rank,_ = compute_analytical_errors_criteria(dataset,num_refst,n_empty,s,alphas[num_refst],save_results=False)
        mse_Dopt[num_refst] = result_Dopt
        mse_rank[num_refst] = result_rank
        
    
    return mse_Dopt, mse_rank
        

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
    
def local_optimization_swap(dataset,n,n_refst,n_lcs,n_empty,s,var,solution_dist,solution_weights,n_swaps=100):
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
    solution_weights : list
        weights on every location for each class of sensor [ [lcs],[refst] ]
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
    
    # sensor palcement failure
    if solution_weights[0].sum()==0 and solution_weights[1].sum()==0:
        print(f'Failure solving sensor placement: reference stations: {n_refst}\n unmonitored locations: {n_empty}\n variances ratio: {var:.2e}')
        return solution_dist, np.inf
    
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
            print(f'Swapping indices:\n idx RefSt: {i}\n idx unmonitored: {j}')
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
                print(f'New distribution: {sensor_placement.locations}')
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

# =============================================================================
# Plots
# =============================================================================

def scientific_notation(x, ndp,show_prefix=False):
    s = '{x:0.{ndp:d}e}'.format(x=x, ndp=ndp)
    m, e = s.split('e')
    if show_prefix:
        return r'{m:s}\times 10^{{{e:d}}}'.format(m=m, e=int(e))
    else:
        return r'10^{{{e:d}}}'.format(m=m, e=int(e))
    
class Plots():
    def __init__(self,save_path,figx=3.5,figy=2.5,fs_title=10,fs_label=10,fs_ticks=10,fs_legend=10,marker_size=3,dpi=300,show_plots=False):
        self.figx = figx
        self.figy = figy
        self.fs_title = fs_title
        self.fs_label = fs_label
        self.fs_ticks = fs_ticks
        self.fs_legend = fs_legend
        self.marker_size = marker_size
        self.dpi = dpi
        self.save_path = save_path
        if show_plots:
            self.backend = 'Qt5Agg'
        else:
            self.backend = 'Agg'
        
        print('Setting mpl rcparams')
        
        font = {'weight':'normal',
                'size':str(self.fs_label),
                }
        
        lines = {'markersize':self.marker_size}
        
        fig = {'figsize':[self.figx,self.figy],
               'dpi':self.dpi
               }
        
        ticks={'labelsize':self.fs_ticks
            }
        axes={'labelsize':self.fs_ticks,
              'grid':True,
              'titlesize':self.fs_title
            }
        
        grid = {'alpha':0.5}
        mpl.rc('grid',**grid)
    
        mathtext={'default':'regular'}
        legend = {'fontsize':self.fs_legend}
        
        
        
        mpl.rc('font',**font)
        mpl.rc('figure',**fig)
        mpl.rc('xtick',**ticks)
        mpl.rc('ytick',**ticks)
        mpl.rc('axes',**axes)
        mpl.rc('legend',**legend)
        mpl.rc('mathtext',**mathtext)
        mpl.rc('lines',**lines)
        
        mpl.use(self.backend)
        
    def ranking_error_comparison(self,errors_sorted_zero,locations_sorted_zero,errors_sorted,Dopt_error,variance_Dopt,idx_rankMax,idx_Dopt,n_refst,s,save_fig=False):
        """
        Plot ranking-sorted locations from lowest to highest RMSE

        Parameters
        ----------
        errors_sorted_zero : numpy array
            sorted RMSE at variance == 0
        locations_sorted_zero : numpy array
            index of locations sorted according to their RMSE
        errors_sorted : numpy array
            errors at different variance but sorted according to ranking for variance == 0
        Dopt_error : float
            RMSE Doptimal method obtained for var!=0
        variance_Dopt: float
            Variance used for obtaining solution for Dopt
        idx_rankMax : int
            index of rankMax solution within all possible configurations for variance == 0
        idx_Dopt : int
            index of Dopt solution (var!=0) within all possible configurations for variance == 0
        n_refst : int
            number of reference stations in the network
        s : int
            signal sparsity
        save_fig : bool, optional
            Save generated figure. The default is False.

        Returns
        -------
        None.

        """
     
        # Indices at criteria results
        xrange = np.arange(1,len(errors_sorted_zero)+1,1)
        rank_loc = np.argwhere(locations_sorted_zero == idx_rankMax)[0][0]
        
        
        try:
            Dopt_loc = np.argwhere(locations_sorted_zero == idx_Dopt)[0][0]
        except:
            Dopt_loc = np.inf
            
            
        fig = plt.figure()
        ax = fig.add_subplot(111)
       
            
        # exhaustive ranking for var!=0
        if var_Dopt != 1e-6:
            ax.plot(xrange,errors_sorted,
                    color='#117a65',label=r'$\epsilon^2/\sigma^2=$'r'${0:s}$'.format(scientific_notation(var_Dopt, 1)),alpha=0.7)
            
        # exhaustive results for var==0
        ax.plot(xrange,errors_sorted_zero,color='k',label=r'$\epsilon^2/\sigma^2=0.0$',alpha=1.0)
        # highlight locations of different criteria
        ax.vlines(x=rank_loc,ymin = 0.0, ymax = np.max(errors_sorted_zero),colors='orange',label='rankMax')
        ax.scatter(x=rank_loc,y=errors_sorted_zero[rank_loc],color='orange')
        
    
        if Dopt_error != np.inf:
            ax.vlines(x=Dopt_loc,ymin = 0.0, ymax = np.max(errors_sorted_zero),colors='#1a5276',label='HJB')
            ax.scatter(x=Dopt_loc,y=errors_sorted_zero[Dopt_loc],color='#1a5276')
            
        
        ax.set_yscale('log')
        ax.set_yticks(np.logspace(-1,1,3))
        ax.set_ylabel('RMSE')
        ax.set_ylim(1e-1,1e1)
        idx = [int(i) for i in np.logspace(0,4,5)]
        ax.set_xticks(xrange[idx])
        ax.set_xticklabels([r'${0:s}$'.format(scientific_notation(i, 1)) for i in ax.get_xticks()])
        ax.set_xscale('log')
        ax.set_xlabel(r'$i$-th configuration')
        ax.legend(loc='upper center',ncol=3,bbox_to_anchor=(0.5, 1.15),framealpha=1)
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        
        if save_fig:
            fname = f'{self.save_path}RMSE_rankingComparison_{n_refst}RefSt_var{var}_varDopt{var_Dopt}_r{s}.png'
            fig.savefig(fname,dpi=300,format='png')
            
    def histogram_error(self,n,s,n_refst,n_empty,var,var_Dopt,errors_sorted,rmse_Dopt,rmse_rankMax,save_fig):
        bins = np.arange(0,1e1+1e-1,1e-1)
        rmse_mean = errors_sorted.mean()
        rmse_std = errors_sorted.std()
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(errors_sorted,bins,density=True,color='#148f77')
        ax.vlines(x=rmse_mean,ymin=0.0,ymax = 3,color='#b03a2e',label='mean RMSE',linestyles='dashed',linewidth=1)
        ax.axvspan(xmin=rmse_mean - rmse_std, xmax = rmse_mean + rmse_std,ymin=0.0,ymax=3,color='#cb4335',alpha=0.2)
        ax.vlines(x=rmse_rankMax,ymin=0.0,ymax = 3,color='orange',label='rankMax',linestyles='dashed',linewidth=1)
        if rmse_Dopt !=np.inf:
            ax.vlines(x=rmse_Dopt,ymin=0.0,ymax = 3,color='k',label='HJB',linestyles='dashed',linewidth=1)
        
        ax.grid(False)
        yrange = np.arange(0.,3+0.5,0.5)
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.round(i,1) for i in ax.get_yticks()])
        ax.set_ylabel('Probability density')
        ax.set_ylim(0,3)
        
        xrange = np.arange(0,10+0.5,0.5)
        ax.set_xticks(xrange)
        ax.set_xticklabels([np.round(i,2) for i in ax.get_xticks()],rotation=45)
        ax.set_xlabel('RMSE ($\mu g/m^{3}$)')
        ax.set_xlim(0,10)
        
        ax.legend(loc='upper right',ncol=1,framealpha=1)
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()

        if save_fig:
            fname = f'{self.save_path}RMSE_histogram_N{n}_{n_refst}RefSt_Empty{n_empty}_r{s}_var{var}_varDopt{var_Dopt}.png'
            fig.savefig(fname,dpi=300,format='png')

        
#%%
if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    exhaustive_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/Exhaustive_network/'
    
    # network paramteres
    n = 18
    n_refst = 3
    n_lcs = 2
    n_empty = 13
    s = 5#n_refst + n_lcs
    
    n_refst_range = np.arange(0,s,1)
    
    # load or generate exhaustive sensors distribution
    exhaustive_placement = ExhaustivePlacement(n, n_refst, n_lcs, n_empty)
    get_exhaustive_distribution(exhaustive_placement, exhaustive_path)
    exhaustive_placement.sort_locations()
    
    # get reduced dataset
    dataset = load_dataset()
    
    #%% solve convex sensor placement problem
    solve = False
    if solve:
        criterion = 'rankMax' #['rankMax','D_optimal']
        solve_sensor_placement(dataset,criterion)
        sys.exit()
    #%% validate alpha rankMax
    validate = False
    if validate:
        
        print(f'Validating alpha value for rankMax results.\nN: {n}\n sparsity: {s}\n ')
        input('Press enter to continue ...')
    
        df_alphas = pd.DataFrame()
        df = pd.DataFrame(data=None,index=n_refst_range,columns=[n_empty])
        for n_refst in n_refst_range:
            mse_alphas,alpha_min = validation_alpha(dataset,n,n_refst=n_refst,n_empty=n_empty,s=s,
                                          files_path=exhaustive_path,criterion='rankMax')
            
            df.loc[n_refst,n_empty] = alpha_min
        df_alphas = pd.concat((df_alphas,df),axis=1)
    
        df_alphas.to_csv(results_path+f'Validation_results_{n}N_{s}r_{n_empty}nEmpty.csv')
        sys.exit()
    #%%
    # compute MSE using both criteria and exhaustive configurations for given variance
    estimate = False
    if estimate:
       
        print(f'Exhaustive estimation\n N:{n}\n n_refst: {n_refst}\n n_empty: {n_empty}\n s:{s}')
        input('Print Enter to continue ...')
        #df_alphas = pd.read_csv(exhaustive_path+f'Validation_results_{n}N_{s}r_{n_empty}nEmpty.csv',index_col=0)
        #alpha_reg = df_alphas.loc[n_refst,str(n_empty)]
        #mse_Dopt,mse_rank = compute_analytical_errors_criteria(dataset,n_refst,n_empty,s,alpha_reg,save_results=True)
        
        
        variances = [0.0,1e-6,1e-2]#np.concatenate(([0.0],np.logspace(-6,0,7)))
        num_el_chunk = 5000
        num_files = int(np.ceil(exhaustive_placement.num_distributions/num_el_chunk))
        for var in variances:
        # exhaustive estimation: splitted into chunks
            estimation_exhaustive = compute_analytical_errors_exhaustive(dataset, n_refst, var,num_el_chunk,save_results=True)
            
            join_exhaustive_placement_iterations(results_path,var,num_files,locations='Unmonitored')
            join_exhaustive_placement_iterations(results_path,var,num_files,locations='Full')
            join_exhaustive_placement_iterations(results_path,var,num_files,locations='RefSt')
            join_exhaustive_placement_iterations(results_path,var,num_files,locations='LCS')
            
        sys.exit()
        
     
  
        
    #%%
    ## plot RMSE vs ranking at var==0. Highlight criteria location in the ranking. REPEAT RMSE vs ranking at var==0 for results with var!=0
    show_plot = True
    if show_plot:
        print(f'Generating figure for rankMax and Dopt methods.\n n: {n}\n n_refst: {n_refst}\n n_empty:{n_empty}\n s:{s}')
        input('Press Enter to continue...')
        
        var = 0.0
        var_Dopt = 1e-6
        loc = 'Full' # ['Full','RefSt','LCS','Unmonitored']
        
        # load rankMax locations
        
        df_alphas = pd.read_csv(exhaustive_path+f'Validation_results_{n}N_{s}r_{n_empty}nEmpty.csv',index_col=0)
        alpha_reg = df_alphas.loc[n_refst,str(n_empty)]
        
        fname = exhaustive_path+f'DiscreteLocations_rankMax_vs_p0_{n}N_r{s}_pEmpty{n_empty}_alpha{alpha_reg:.1e}.pkl'
        with open(fname,'rb') as f:
            rankMax_locations = pickle.load(f)
        rankMax_locations_nrefst = rankMax_locations[n_refst]
        
        fname = exhaustive_path+f'Weights_rankMax_vs_p0_{n}N_r{s}_pEmpty{n_empty}_alpha{alpha_reg:.1e}.pkl'
        with open(fname,'rb') as f:
            rankMax_locations = pickle.load(f)
        rankMax_weights_nrefst = rankMax_locations[n_refst]
        
        # load Dopt locations
        fname = exhaustive_path+f'DiscreteLocations_D_optimal_vs_p0_{n}N_r{s}_pEmpty{n_empty}_varZero{var_Dopt:.1e}.pkl'
        with open(fname,'rb') as f:
            Dopt_locations = pickle.load(f)
        Dopt_locations_nrefst = Dopt_locations[n_refst]
        
        fname = exhaustive_path+f'Weights_D_optimal_vs_p0_{n}N_r{s}_pEmpty{n_empty}_varZero{var_Dopt:.1e}.pkl'
        with open(fname,'rb') as f:
            Dopt_locations = pickle.load(f)
        Dopt_weights_nrefst = Dopt_locations[n_refst]
      
        # compute MSE: local optimization swapping
        print('Local optimization: swapping')
        print('rankMax')
        rankMax_locations_swap, rankMax_mse_swap = local_optimization_swap(dataset, n, n_refst, n_lcs, n_empty, 
                                                                           s, var, rankMax_locations_nrefst,rankMax_weights_nrefst)
        
        rankMax_error_swap = np.sqrt(rankMax_mse_swap)
        
        
        print('Doptimal')
        Dopt_locations_swap, Dopt_mse_swap = local_optimization_swap(dataset, n, n_refst, n_lcs, n_empty, 
                                                                           s, var, Dopt_locations_nrefst,Dopt_weights_nrefst)
        Dopt_error_swap = np.sqrt(Dopt_mse_swap)
        
        # load exhaustive results for var==0
        exhaustive_placement.load_errors_exhaustive(loc, var, n_refst,s,exhaustive_path)
        errors_sorted_zero= exhaustive_placement.errors_sorted.copy()
        locations_sorted_zero = exhaustive_placement.loc_sorted.copy()
        
        
        #index of criteria within exhaustive search
        idx_rankMax = exhaustive_placement.find_specific_configuration(rankMax_locations_swap)
        idx_Dopt = exhaustive_placement.find_specific_configuration(Dopt_locations_swap)
        
        
        # load exhaustive results for var!=0 and sort them according to index for var==0
        exhaustive_placement.load_errors_exhaustive(loc, var_Dopt, n_refst,s,exhaustive_path)
        errors_sorted = np.sqrt(np.array([i for i in exhaustive_placement.exhaustive_mse.values()]))[locations_sorted_zero]
        
        print(f'RMSE ratio RMSE_optimal/RMSE_method\n rankMax: {errors_sorted_zero[0] / rankMax_error_swap :.2f}\n Dopt: {errors_sorted_zero[0] / Dopt_error_swap :.2f} ')
        
        # figure
        plots = Plots(save_path=results_path,marker_size=3,
                           fs_label=7,fs_ticks=7,fs_legend=5,fs_title=10,
                           show_plots=True)
        
        # ranking plot
        plots.ranking_error_comparison(errors_sorted_zero,locations_sorted_zero,errors_sorted,
                                       Dopt_error_swap,var_Dopt,idx_rankMax,idx_Dopt,n_refst,s,save_fig=False)
       
        # histogram plot
        plots.histogram_error(n, s, n_refst, n_empty,var,var_Dopt,
                              errors_sorted_zero,Dopt_error_swap,rankMax_error_swap,save_fig=False)
 
  
    