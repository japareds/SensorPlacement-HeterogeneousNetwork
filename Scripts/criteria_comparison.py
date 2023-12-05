#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:09:41 2023

@author: jparedes
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle
import sys
import warnings

import SensorPlacement as SP
import DataSet as DS
import LowRankBasis as LRB
import compute_weights as CW
import Estimation


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
    variances = variances = np.concatenate(([0.0],np.logspace(-6,0,4)))
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
#%%

# =============================================================================
# Plots
# =============================================================================
            
class Plots():
    def __init__(self,save_path,figx=3.5,figy=2.5,fs_title=10,fs_label=10,fs_ticks=10,fs_legend=10,marker_size=3,dpi=300,use_grid=False,show_plots=False):
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
              'grid':False,
              'titlesize':self.fs_title
            }
        if use_grid:
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
        
    def heatmap_criteria_ratio(self,df_rmse_rankMax,df_rmse_Dopt,n,s,var,center_value=1.0,extreme_range=0.5,text_note_size=5,save_fig=False):
        """
        Heatmap image showing RMSE ratio between rankMax solutions and Doptimal solutions.
        If the ratio is less than one then rankMax locations are better for reconstructing the signal.
        
        The image shows RMSE ratio for different number of RefSt: 0 to s-1
        and different number of unmonitored locations: 0(Fully monitored network) to n-s (scarcely monitored network)
        
        The image is center to 1.0 (equal error) and ranges from 0.5 up to 1.5 times for comparison with other variance ratios
        or network configurations.

        Parameters
        ----------
        df_rmse_rankMax : pandas dataframe
            rankMax RMSE
        df_rmse_Dopt : pandas dataframe
            Doptimal RMSE
        n : int
            netwrok size
        s : int
            sparsity level
        var : float
            variances ratio
        text_note_size : int
            size of text showing pixel value
        save_fig : bool, optional
            save generated figure. The default is False.

        Returns
        -------
        fig : matplotlib figure
            image of RMSE ratios

        """
        
        df_rmse_ratio = df_rmse_rankMax.astype(float)/df_rmse_Dopt.astype(float)
        # zero-valued ratio is might happen because df_rmse_Dopt diverges and not because df_rmse_rank is zero
        df_rmse_ratio.replace(0,np.nan,inplace=True)
        
        min_val = df_rmse_ratio.to_numpy().min()
        max_val = df_rmse_ratio.to_numpy().max()
        print(f'Maximum ratio: {max_val:.2f}\nMinimum ratio: {min_val:.2f}')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        crange = np.arange(center_value-extreme_range,center_value+extreme_range+0.1,0.1)
        cmap = mpl.colormaps['summer'].resampled(64)#len(crange)
        cmap.set_bad('k',0.8)
        
        # different colorbar normalizations choose one
        bnorm = mpl.colors.BoundaryNorm(crange, cmap.N,extend='max')
        cnorm = mpl.colors.CenteredNorm(vcenter=center_value,halfrange=extreme_range)
        divnorm = mpl.colors.TwoSlopeNorm(vcenter=1.0,vmin=0.5,vmax=np.ceil(max_val))
        
        im = ax.imshow(df_rmse_ratio.T,cmap,
                       norm=cnorm,interpolation=None)
        
        cbar_extension = 'both' if (center_value-extreme_range) != 0 else 'neither'
        
        cbar = fig.colorbar(im,ax=ax,
                            orientation='horizontal',location='top',
                            label=r'$RMSE_{RM}$ / $RMSE_{HJB}$',extend=cbar_extension)
        
        if text_note_size != 0:
            for (j,i),label in np.ndenumerate(df_rmse_ratio.T):
                ax.text(i,j,f'{label:.2f}',ha='center',va='center',color='k',size=text_note_size)
            

        
        
        cbar.set_ticks(np.arange(center_value-extreme_range,center_value+extreme_range+0.25,0.25))
        cbar.set_ticklabels([round(i,2) for i in cbar.get_ticks()])
        
        xrange = [i for i in df_rmse_ratio.T.columns]
        ax.set_xticks(np.arange(xrange[0],xrange[-1],5))
        ax.set_xticklabels(ax.get_xticks())
        ax.set_xlabel('Number of\n reference stations')
        
        yrange = [int(i) for i in df_rmse_ratio.columns]
        ax.set_yticks(np.arange(yrange[0],yrange[-1],5))
        ax.set_yticklabels(ax.get_yticks())
        ax.set_ylabel('Number of\n unmonitored locations')
        
        ax.set_aspect('auto')
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        if save_fig:
            fname = self.save_path+f'RMSEratio_RefSt_vs_Unmonitored_r{s}_N{n}_var{var:.2e}.png'
            fig.savefig(fname,dpi=300,format='png')
        
        return fig
       
#%%
if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
    
    #exhaustive_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/Exhaustive_network/'
    
    # network paramteres
    N = 18 #[18,71]
    POLLUTANT = 'O3' #['O3','NO2']
    
    if N==71 and POLLUTANT == 'O3':
        S = 50
    elif N==71 and POLLUTANT == 'NO2':
        S = 53
    elif N==18 and POLLUTANT == 'O3':
        S = 5
    
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
        #input('Press Enter to continue...')
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
    estimate = False
    if estimate:
        var = 0e0
        if var == 0.0:
            var_Dopt = 1e-6
        else:
            var_Dopt = var
        
        df_alphas = pd.read_csv(results_path+f'Criteria_comparison/Validation_results_{N}N_{S}r.csv',index_col=0)
        
        df_rmse_rankMax = pd.DataFrame()
        df_rmse_Dopt = pd.DataFrame()
        
        print(f'Estimating RMSE using both criteria solutions.\n Pollutant: {POLLUTANT}\n N: {N}\n sparsity: {S}\n Variances ratio: {var:.1e}\n Number of unmonitored locations ranges from {n_empty_range[0]} to {n_empty_range[-1]}\n Number of reference stations ranges from 0 to {S-1}\n The rest are LCS up to complete monitored locations')
        input('Press Enter to continue ...')
        for n_empty in n_empty_range:
            print(f'{n_empty} unmonitored locations')
            df_rankMax = pd.DataFrame(data=None,index = n_refst_range,columns=[n_empty])
            df_Dopt = pd.DataFrame(data=None,index = n_refst_range,columns=[n_empty])
            for n_refst in n_refst_range:
                print(f'{n_refst} reference stations')
                # load rankMax and Dopt locations
                alpha_reg = df_alphas.loc[n_refst,str(n_empty)]
                
                fname = results_path+f'Criteria_comparison/DiscreteLocations_rankMax_vs_p0_{N}N_r{S}_pEmpty{n_empty}_alpha{alpha_reg:.1e}.pkl'
                with open(fname,'rb') as f:
                    rankMax_locations = pickle.load(f)
                rankMax_locations_nrefst = rankMax_locations[n_refst]
                
                fname = results_path+f'Criteria_comparison/Weights_rankMax_vs_p0_{N}N_r{S}_pEmpty{n_empty}_alpha{alpha_reg:.1e}.pkl'
                with open(fname,'rb') as f:
                    rankMax_locations = pickle.load(f)
                rankMax_weights_nrefst = rankMax_locations[n_refst]
                
                
                fname = results_path+f'Criteria_comparison/DiscreteLocations_D_optimal_vs_p0_{N}N_r{S}_pEmpty{n_empty}_varZero{var_Dopt:.1e}.pkl'
                with open(fname,'rb') as f:
                    Dopt_locations = pickle.load(f)
                Dopt_locations_nrefst = Dopt_locations[n_refst]
                
                fname = results_path+f'Criteria_comparison/Weights_D_optimal_vs_p0_{N}N_r{S}_pEmpty{n_empty}_varZero{var_Dopt:.1e}.pkl'
                with open(fname,'rb') as f:
                    Dopt_locations = pickle.load(f)
                Dopt_weights_nrefst = Dopt_locations[n_refst]
                
        
                # local optimization : swapping
                print('Local optimization: swapping')
                print('rankMax')
                rankMax_locations_swap, rankMax_mse_swap = local_optimization_swap(dataset, N, n_refst, N-n_refst-n_empty, n_empty, 
                                                                                   S, var, rankMax_locations_nrefst,rankMax_weights_nrefst)
                
                rankMax_error_swap = np.sqrt(rankMax_mse_swap)
                
                
                print('Doptimal')
                Dopt_locations_swap, Dopt_mse_swap = local_optimization_swap(dataset, N, n_refst, N-n_refst-n_empty, n_empty, 
                                                                                   S, var, Dopt_locations_nrefst,Dopt_weights_nrefst)
                Dopt_error_swap = np.sqrt(Dopt_mse_swap)
                
                df_rankMax.loc[n_refst,n_empty] = rankMax_error_swap
                df_Dopt.loc[n_refst,n_empty] = Dopt_error_swap
            
            df_rmse_rankMax = pd.concat((df_rmse_rankMax,df_rankMax),axis=1)
            df_rmse_Dopt = pd.concat((df_rmse_Dopt,df_Dopt),axis=1)
            
        
        
        df_rmse_rankMax.to_csv(results_path+f'RMSE_rankMax_{N}N_{S}r_var{var:.2e}.csv')
        df_rmse_Dopt.to_csv(results_path+f'RMSE_Dopt_{N}N_{S}r_var{var:.2e}.csv')
        sys.exit()
        
    show_plots = True
    if show_plots:
        # load files with RMSE for both criteria
        var = 0e0 #[1e0,1e-2,1e-4,1e-6]
        print(f'Showing figures comparing both sensor placement criteria\n Network size: {N}\n sparsity: {S}\n variances ratio: {var:.2e}')
        input('Press Enter to continue ...')
        df_rmse_rankMax = pd.read_csv(results_path+f'Criteria_comparison/RMSE_rankMax_{N}N_{S}r_var{var:.2e}.csv',index_col=0)
        df_rmse_Dopt = pd.read_csv(results_path+f'Criteria_comparison/RMSE_Dopt_{N}N_{S}r_var{var:.2e}.csv',index_col=0)
        
        plots = Plots(save_path=results_path,marker_size=1,
                            fs_label=7,fs_ticks=7,fs_legend=5,fs_title=10,
                            show_plots=True)
        plots.heatmap_criteria_ratio(df_rmse_rankMax,df_rmse_Dopt,
                                     N,S,var,
                                     center_value=0.5,extreme_range=0.5,
                                     text_note_size=5,save_fig=False)

    
   