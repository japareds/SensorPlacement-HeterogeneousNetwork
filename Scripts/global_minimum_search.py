#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 10:57:41 2023

@author: jparedes
"""


import itertools
import os
import pandas as pd
import numpy as np
import math
import warnings
import pickle

import argparse

import DataSet as DS
import LowRankBasis as LRB
import Estimation
#%% Script inoput parameters
parser = argparse.ArgumentParser(prog='Global_mminimum_search',
                                 description='Exhaustive search over all possible combinations to find the global minimum sensors distributions',
                                 epilog='---')
parser.add_argument('-nr','--n_refst',help='Number of reference stations in the network',type=int,required=True)
parser.add_argument('-ne','--n_empty',help='Number of unmonitored locations in the network',type=int,required=True)
parser.add_argument('--estimate', help='Flag for computing RMSE for all possible combinations',action='store_true',default='True')
args = parser.parse_args()

#%% Exhaustive placement class

        
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
            

#%% compute errors

def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k:data[k] for k in itertools.islice(it, SIZE)}

def compute_minimum_analytical_rmse(dataset,lowrank_basis,exhaustive_placement,n,n_refst,n_empty,s,var,num_el_chunk = 5000):
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
    it = 0
    min_mse = []
    locations_min = []
    
    for item in chunks(exhaustive_placement.locations, num_el_chunk): # split dict into chunks for memory
        new_locs = {el:i for el,i in zip(range(num_el_chunk),item.values())}    
        estimation_exhaustive = Estimation.Estimation(n, s, 
                                                      n_empty, n_refst, 
                                                      1e0, var, 
                                                      1, 1e0,
                                                      [], [], [],
                                                      lowrank_basis.Psi, '','')
        estimation_exhaustive.analytical_estimation_exhaustive(new_locs)
        
        # if save_results:
        #     fname = f'{results_path}MSE_analytical_exhaustive_Full_{n_refst}RefSt_var{var:.1e}_r{s}_it{it}.pkl'
        #     with open(fname,'wb') as f:
        #        pickle.dump(estimation_exhaustive.mse_analytical_full,f)
               
        #     fname = f'{results_path}MSE_analytical_exhaustive_Unmonitored_{n_refst}RefSt_var{var:.1e}_r{s}_it{it}.pkl'
        #     with open(fname,'wb') as f:
        #        pickle.dump(estimation_exhaustive.mse_analytical_unmonitored,f)
            
        #     fname = f'{results_path}MSE_analytical_exhaustive_RefSt_{n_refst}RefSt_var{var:.1e}_r{s}_it{it}.pkl'
        #     with open(fname,'wb') as f:
        #        pickle.dump(estimation_exhaustive.mse_analytical_refst,f)
            
        #     fname = f'{results_path}MSE_analytical_exhaustive_LCS_{n_refst}RefSt_var{var:.1e}_r{s}_it{it}.pkl'
        #     with open(fname,'wb') as f:
        #        pickle.dump(estimation_exhaustive.mse_analytical_lcs,f)
        
        # keep minimum error
        min_mse.append(np.min([i for i in estimation_exhaustive.mse_analytical_full.values()]))
        # keep distribution of minimum error
        idx_min = np.argmin([i for i in estimation_exhaustive.mse_analytical_full.values()])
        locations_min.append(new_locs[idx_min])
        
        
        it+=1
    
    min_rmse = np.sqrt(np.min(min_mse))
    idx_min_rmse = np.argmin(min_mse)
    location_min_rmse = locations_min[idx_min_rmse]
    return  min_rmse, location_min_rmse

def join_exhaustive_placement_iterations(path_files,n_refst,s,var,n_it,locations='Unmonitored'):
    
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

        
def get_exhaustive_distribution(exhaustive_placement,exhaustive_path):
    try:
        exhaustive_placement.load_locations(exhaustive_path)
    except:
        exhaustive_placement.distribute()
        exhaustive_placement.save_locations(results_path)
    
#%% dataset
def load_dataset(pollutant,n,files_path):
    
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
    val_set_end = '2022-02-19 02:00:00'#'2022-12-31 23:00:00'
    test_set_begin = '2022-02-19 03:00:00'#'2022-01-01 00:00:00'
    
    dataset.train_test_split(train_set_end, val_set_begin, val_set_end,test_set_begin)
    
    return dataset

#%% loop
def exhaustive_loop(M,S,POLLUTANT,var):
    # range of reference stations and unmonitored locations
    
    n_refst_range = np.arange(0,S,1)
    n_empty_range = np.arange(0,N-S+1,1)
    
    print(f'Searching for minimum RMSE\n Pollutant: {POLLUTANT}\n N: {N}\n sparsity: {S}\n Variances ratio: {var:.1e}\n Number of unmonitored locations ranges from {n_empty_range[0]} to {n_empty_range[-1]}\n Number of reference stations ranges from 0 to {S-1}\n The rest are LCS up to complete monitored locations')
    input('Press Enter to continue ...')
    df_rmse_min = pd.DataFrame()
    
    for n_empty in n_empty_range:
        print(f'{n_empty} unmonitored locations')
        df_exhaustive_placement = pd.DataFrame(data=None,index = n_refst_range,columns=[n_empty])
        for n_refst in n_refst_range:
            n_lcs = N - n_refst - n_empty
            print(f'{n_refst} reference stations\n{n_lcs} LCSs')
            
            # get all possible configurations
            exhaustive_placement = ExhaustivePlacement(N, n_refst, n_lcs, n_empty)
            exhaustive_placement.distribute()
            exhaustive_placement.sort_locations()
            
            # estimate RMSE for all configurations
            num_el_bins = 5000
            num_files = int(np.ceil(exhaustive_placement.num_distributions/num_el_bins))
            
            min_rmse = compute_minimum_analytical_rmse(dataset, lowrank_basis, exhaustive_placement, 
                                                      N, n_refst, n_empty, 
                                                      S, var,num_el_bins)
            
            df_exhaustive_placement.loc[n_refst,n_empty] = min_rmse
            
            
        df_rmse_min = pd.concat((df_rmse_min,df_exhaustive_placement),axis=1)
    
    df_rmse_min.to_csv(results_path+f'RMSE_globalMin_{N}N_{S}r_var{var:.2e}.csv')
    
    return df_rmse_min

def load_globalMin_files(N,S,var,path):
    n_refst_range = np.arange(0,S,1)
    n_empty_range = np.arange(0,N-S+1,1)
    
    df_rmse_min = pd.DataFrame()
    for n_empty in n_empty_range:
        print(f'loading data for {n_empty} unmonitored locations')
        df_refst = pd.DataFrame()
        for n_refst in n_refst_range:
            print(f'loading data for {n_refst} reference stations')
            fname = path+f'RMSE_globalMin_Refst{n_refst}_Unmonitored{n_empty}_N{N}_{S}r_var{var:.2e}.csv'
            df = pd.read_csv(fname,index_col=0)
            df_refst = pd.concat((df_refst,df),axis=0)
        df_rmse_min = pd.concat((df_rmse_min,df_refst),axis=1)
    
    return df_rmse_min
            
    
#%%
if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
   
    # network paramteres
    N = 18
    S = 5
    POLLUTANT = 'O3' #['O3','NO2']
    var = 0e0
    
    dataset = load_dataset(POLLUTANT,N,files_path)
    # project dataset onto subspace so that data is exactly sparse in a basis
    lowrank_basis = LRB.LowRankBasis(dataset.ds_test,S)
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    dataset.project_basis(lowrank_basis.Psi)
    
    
    
    if args.estimate:
        print(f'Searching for minimum RMSE\n Pollutant: {POLLUTANT}\n N: {N}\n sparsity: {S}\n Variances ratio: {var:.1e}')
        df_rmse_min = pd.DataFrame(data=None,index=[args.n_refst],columns=[args.n_empty])
        
        n_lcs = N - args.n_refst - args.n_empty
        
        print(f'{args.n_empty} unmonitored locations')
        print(f'{args.n_refst} reference stations\n{n_lcs} LCSs')
        
        # get all possible configurations
        exhaustive_placement = ExhaustivePlacement(N, args.n_refst, n_lcs, args.n_empty)
        exhaustive_placement.distribute()
        exhaustive_placement.sort_locations()
        
        # estimate RMSE for all configurations
        num_el_bins = 5000
        num_files = int(np.ceil(exhaustive_placement.num_distributions/num_el_bins))
        
        min_rmse, location_min_rmse = compute_minimum_analytical_rmse(dataset, lowrank_basis, exhaustive_placement, 
                                                  N, args.n_refst, args.n_empty, 
                                                  S, var,num_el_bins)
        
        df_rmse_min.loc[args.n_refst,args.n_empty] = min_rmse
        print(f'Minimum RMSE found: {min_rmse}')
        print(f'Optimal distribution: {location_min_rmse}')
        
        df_rmse_min.to_csv(results_path+f'RMSE_globalMin_RefSt{args.n_refst}_Unmonitored{args.n_empty}_N{N}_{S}r_var{var:.2e}.csv')
        
        fname = f'{results_path}DiscreteLocations_globalMin_RefSt{args.n_refst}_Unmonitored{args.n_empty}_N{N}_{S}r_var{var:.2e}.pkl'
        with open(fname,'wb') as f:
            pickle.dump(location_min_rmse,f)
    
    else:
        print(f'Generating data set for all possible combinations\nObtaining data files from: {results_path}')
        df_rmse_min = load_globalMin_files(N,S,var,results_path)
        fname = f'RMSE_globalMin_{POLLUTANT}_{N}N_{S}r_var{var:.2e}.csv'
        df_rmse_min.to_csv(results_path+fname)
           