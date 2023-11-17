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

import DataSet as DS
import LowRankBasis as LRB
import compute_weights as CW
import Estimation
import Plots


#%% locations algorithms
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
    
def validation_alpha(dataset,n_refst):

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
        estimation_rankMax.analytical_estimation(criterion='rankMax')
        
        mse_alphas[alpha_reg] = estimation_rankMax.mse_analytical_full
        
    return mse_alphas

def compute_analytical_errors_criteria(dataset,n_refst,alpha_reg,save_results=False):
    """
    Compute analytical MSE from trace covariance matrix for multiple number of reference stations in the network and 
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
    lowrank_basis = LRB.LowRankBasis(dataset.ds_train,s)
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    dataset.project_basis(lowrank_basis.Psi)
    
    variances = np.concatenate(([0.0],np.logspace(-6,0,7)))
        
    dict_Dopt_var = {el:np.inf for el in variances}
    dict_rankMax_var = {el:np.inf for el in variances}
    dict_ex_var = {el:np.inf for el in variances}
        
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
        
        estimation_exhaustive = Estimation.Estimation(n, s, n_empty, 
                                                n_refst, 1e0, var, 
                                                1, alpha_reg,
                                                [], [], [],
                                                lowrank_basis.Psi, exhaustive_path,exhaustive_path)
        
        
        if var!= 0:
        
            estimation_Dopt.analytical_estimation(criterion='D_optimal')
            dict_Dopt_var[var] = [estimation_Dopt.mse_analytical_full,estimation_Dopt.mse_analytical_refst,estimation_Dopt.mse_analytical_lcs,estimation_Dopt.mse_analytical_unmonitored]
            
            estimation_rankMax.analytical_estimation(criterion='rankMax')
            dict_rankMax_var[var] = [estimation_rankMax.mse_analytical_full,estimation_rankMax.mse_analytical_refst,estimation_rankMax.mse_analytical_lcs,estimation_rankMax.mse_analytical_unmonitored]
            
            # estimation_exhaustive.analytical_estimation_exhaustive(exhaustive_placement)
            # dict_ex_var[var] = [estimation_exhaustive.mse_analytical_full,estimation_exhaustive.mse_analytical_refst,
            #                     estimation_exhaustive.mse_analytical_lcs,estimation_exhaustive.mse_analytical_unmonitored]
            
        else:
            dict_Dopt_var[var] = [np.inf,np.inf,np.inf,np.inf]
            
            estimation_rankMax.analytical_estimation(criterion='rankMax')
            dict_rankMax_var[var] = [estimation_rankMax.mse_analytical_full,estimation_rankMax.mse_analytical_refst,estimation_rankMax.mse_analytical_lcs,estimation_rankMax.mse_analytical_unmonitored]
            
            # estimation_exhaustive.analytical_estimation_exhaustive(exhaustive_placement)
            # dict_ex_var[var] = [estimation_exhaustive.mse_analytical_full,estimation_exhaustive.mse_analytical_refst,
            #                     estimation_exhaustive.mse_analytical_lcs,estimation_exhaustive.mse_analytical_unmonitored]
            
    mse_Dopt = pd.DataFrame(dict_Dopt_var,index=['Full','RefSt','LCS','Unmonitored'],columns=variances).T
    mse_rank = pd.DataFrame(dict_rankMax_var,index=['Full','RefSt','LCS','Unmonitored'],columns=variances).T
    # mse_ex = pd.DataFrame(dict_ex_var,index=['Full','RefSt','LCS','Unmonitored'],columns=variances).T
    
    
    if save_results:
        fname = f'{results_path}MSE_analytical_Doptimal.pkl'
        with open(fname,'wb') as f:
           pickle.dump(mse_Dopt,f)
        
        fname = f'{results_path}MSE_analytical_rankMax.pkl'
        with open(fname,'wb') as f:
           pickle.dump(mse_rank,f)
           
        # fname = f'{results_path}MSE_analytical_exhaustive.pkl'
        # with open(fname,'wb') as f:
        #    pickle.dump(mse_ex,f)
          
    return mse_Dopt,mse_rank
        

def compute_analytical_errors_exhaustive(dataset,n_refst,var,save_results=False):
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
    input('Compute analytical RMSE using covariance matrices.\nComputing ALL possible locations!\nPress Enter to continue ...')
    # project dataset onto subspace so that data is exactly sparse in a basis
    lowrank_basis = LRB.LowRankBasis(dataset.ds_train,s)
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    dataset.project_basis(lowrank_basis.Psi)
    
        
        
    estimation_exhaustive = Estimation.Estimation(n, s, 
                                                  n_empty, n_refst, 
                                                  1e0, var, 
                                                  1, 1e0,
                                                  [], [], [],
                                                  lowrank_basis.Psi, exhaustive_path,exhaustive_path)
    
    estimation_exhaustive.analytical_estimation_exhaustive(exhaustive_placement)
        
    
    
    if save_results:
        fname = f'{results_path}MSE_analytical_exhaustive_Full_var{var:.1e}.pkl'
        with open(fname,'wb') as f:
           pickle.dump(estimation_exhaustive.mse_analytical_full,f)
           
        fname = f'{results_path}MSE_analytical_exhaustive_Unmonitored_var{var:.1e}.pkl'
        with open(fname,'wb') as f:
           pickle.dump(estimation_exhaustive.mse_analytical_unmonitored,f)
        
        fname = f'{results_path}MSE_analytical_exhaustive_RefSt_var{var:.1e}.pkl'
        with open(fname,'wb') as f:
           pickle.dump(estimation_exhaustive.mse_analytical_refst,f)
        
        fname = f'{results_path}MSE_analytical_exhaustive_LCS_var{var:.1e}.pkl'
        with open(fname,'wb') as f:
           pickle.dump(estimation_exhaustive.mse_analytical_lcs,f)
        
    return estimation_exhaustive

def figure_exhaustive_criteria(errors_sorted,rank_error,Dopt_error,var,save_fig=False):
    plots = Plots.Plots(save_path=results_path,marker_size=1,
                        fs_label=7,fs_ticks=7,fs_legend=5,fs_title=10,
                        show_plots=True)
    
    plots.plot_ranking_error(errors_sorted, rank_error, Dopt_error, var,save_fig)
    
   


        
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
    s = n_refst + n_lcs
    
    # load or generate exhaustive sensors distribution
    exhaustive_placement = ExhaustivePlacement(n, n_refst, n_lcs, n_empty)
    get_exhaustive_distribution(exhaustive_placement, exhaustive_path)
    exhaustive_placement.sort_locations()
    
    # get reduced dataset
    dataset = load_dataset()
    
    # solve convex sensor placement problem
    solve = False
    if solve:
        solve_sensor_placement(dataset,criterion='rankMax')
    # validate alpha rankMax
    validate = False
    if validate:
        mse_alphas = validation_alpha(dataset, n_refst)
    
    # compute MSE using both criteria and exhaustive configurations forn given variance
    estimate = True
    if estimate:
       #mse_Dopt,mse_rank = compute_analytical_errors_criteria(dataset,n_refst,alpha_reg=1e0,save_results=True)
       estimation_exhaustive = compute_analytical_errors_exhaustive(dataset, n_refst, var=1e0)
       
     
    # plot comparison criteria location in the exhaustive ranks
    show_figures = False
    if show_figures:
        # load exhaustive results
        loc = 'Unmonitored'
        var = 0.0
        fname = exhaustive_path+f'MSE_analytical_exhaustive_{loc}_var{var:.1e}.pkl'
        with open(fname,'rb') as f:
            exhaustive_mse = pickle.load(f)
        errors_sorted = np.sqrt(np.sort([i[0] for i in exhaustive_mse.values()]))
        # load criteria results
        
        fname = exhaustive_path+'MSE_analytical_rankMax.pkl'
        with open(fname,'rb') as f:
            rankMax_mse = pickle.load(f)
        rankMax_error = np.sqrt(rankMax_mse.loc[var][loc])
            
        fname = exhaustive_path+'MSE_analytical_Doptimal.pkl'
        with open(fname,'rb') as f:
            Dopt_mse = pickle.load(f)
        Dopt_error = np.sqrt(Dopt_mse.loc[var][loc])
        
        figure_exhaustive_criteria(errors_sorted,rankMax_error,Dopt_error,var,save_fig=True)
        
        
        
       
    
    
    