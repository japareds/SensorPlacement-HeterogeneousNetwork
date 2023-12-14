#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 12:57:50 2023

@author: jparedes
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from geopandas import GeoDataFrame

import matplotlib.pyplot as plt
import matplotlib as mpl

import pickle
import sys
import warnings

import DataSet as DS

#%% dataSet and network
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

def get_locations(files_path,pollutant):
    dataset_source='taiwan'
    if dataset_source =='taiwan':
        start_date = '2018'
        end_date = '2022'
        fname = f'{files_path}/Taiwan/Coordinates_stations_{pollutant}_Taiwan_{end_date}_{start_date}.csv'
    
    with open(fname,'rb') as f:
        coordinates = pickle.load(f)
        
    return coordinates
    
#%% dsitributions
def get_criteria_distributions(n,s,n_refst,n_empty,alpha_reg,var,var_Dopt):
    
    print(f'Loading criteria distributions for solving sensor placement problem.\n N: {n}\n s: {s}\n refst: {n_refst}\n unmonitored: {n_empty}\n var for HJB: {var:.2e}')
    
    # load rankMax locations and weights distribution
    fname = results_path+f'Criteria_comparison/DiscreteLocations_rankMax_vs_p0_{n}N_r{s}_pEmpty{n_empty}_alpha{alpha_reg:.1e}.pkl'
    with open(fname,'rb') as f:
        rankMax_locations = pickle.load(f)
    rankMax_locations_nrefst = rankMax_locations[n_refst]
    
    fname = results_path+f'Criteria_comparison/Weights_rankMax_vs_p0_{n}N_r{s}_pEmpty{n_empty}_alpha{alpha_reg:.1e}.pkl'
    with open(fname,'rb') as f:
        rankMax_locations = pickle.load(f)
    rankMax_weights_nrefst = rankMax_locations[n_refst]
    
    # load Doptimal locations and weights distribution
    fname = results_path+f'Criteria_comparison/DiscreteLocations_D_optimal_vs_p0_{n}N_r{s}_pEmpty{n_empty}_varZero{var_Dopt:.1e}.pkl'
    with open(fname,'rb') as f:
        Dopt_locations = pickle.load(f)
    Dopt_locations_nrefst = Dopt_locations[n_refst]
    
    fname = results_path+f'Criteria_comparison/Weights_D_optimal_vs_p0_{n}N_r{s}_pEmpty{n_empty}_varZero{var_Dopt:.1e}.pkl'
    with open(fname,'rb') as f:
        Dopt_locations = pickle.load(f)
    Dopt_weights_nrefst = Dopt_locations[n_refst]
    
    if Dopt_weights_nrefst[0].sum()== 0.0 and Dopt_weights_nrefst[1].sum()==0.0:
        warnings.warn(f'Doptimal failure\n var: {var:.2e}\n n: {n}\n refst: {n_refst}\n unmonitored: {n_empty}')
        
    # load optimal locations
    fname = results_path+f'Criteria_comparison/DiscreteLocations_globalMin_RefSt{n_refst}_Unmonitored{n_empty}_N{n}_{s}r_var{var:.2e}.pkl'
    with open(fname,'rb') as f:
        optimal_locations = pickle.load(f)
    
    return rankMax_locations_nrefst, Dopt_locations_nrefst, optimal_locations

def get_criteria_rmse(n,s,var,var_Dopt,n_refst,n_empty):
    
    print(f'Loading RMSE obtained by rankMax , HJB and optimal (if exists)\n N: {n}\n s: {s}\n var: {var:.2e}\n refst: {n_refst}\n unmonitored: {n_empty}')
    df_rmse_rankMax = pd.read_csv(results_path+f'Criteria_comparison/RMSE_rankMax_{n}N_{s}r_var{var:.2e}.csv',index_col=0)
    df_rmse_Dopt = pd.read_csv(results_path+f'Criteria_comparison/RMSE_Dopt_{n}N_{s}r_var{var:.2e}_computedVar{var_Dopt:.2e}.csv',index_col=0)
    
    if var == 0.0:
        df_rmse_optimal = pd.read_csv(results_path+f'Criteria_comparison/RMSE_globalMin_{N}N_{S}r_var{var:.2e}.csv',index_col=0)
        rmse_optimal = df_rmse_optimal.loc[n_refst,str(n_empty)]
    else:
        print(f'No optimal search for variance {var:.2e}')
        rmse_optimal = np.nan
        
    rmse_rankMax = df_rmse_rankMax.loc[n_refst,str(n_empty)]
    rmse_Dopt = df_rmse_Dopt.loc[n_refst,str(n_empty)]
    
    return rmse_rankMax, rmse_Dopt, rmse_optimal

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

    def geographical_network_visualization(self,coordinates,criteria_location,criterion,RMSE,n,n_refst,n_empty,s,var,text_size,save_fig=False):
        
        coords_lcs = coordinates[criteria_location[0]]
        coords_refst = coordinates[criteria_location[1]]
        coords_empty = coordinates[criteria_location[2]]
        
        df = pd.DataFrame(coordinates,columns=['Latitude','Longitude'])
        df_lcs = pd.DataFrame(coords_lcs,columns=['Latitude','Longitude'])
        df_refst = pd.DataFrame(coords_refst,columns=['Latitude','Longitude'])
        df_unmonitored =  pd.DataFrame(coords_empty,columns=['Latitude','Longitude'])
        
        geometry_lcs = [Point(xy) for xy in zip(df_lcs['Longitude'], df_lcs['Latitude'])]
        geometry_refst = [Point(xy) for xy in zip(df_refst['Longitude'], df_refst['Latitude'])]
        geometry_empty = [Point(xy) for xy in zip(df_unmonitored['Longitude'], df_unmonitored['Latitude'])]
        
        gdf_lcs = GeoDataFrame(df_lcs, geometry=geometry_lcs)
        gdf_refst = GeoDataFrame(df_refst, geometry=geometry_refst)
        gdf_empty = GeoDataFrame(df_unmonitored, geometry=geometry_empty)
        taiwan = gpd.read_file(f'{files_path}Taiwan/gadm41_TWN.gpkg')
        
        
        #world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        #world_map = world.plot(ax=ax,color='#117a65')
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        taiwan_map = taiwan.plot(ax=ax,color='#117a65')
        
        gdf_empty.plot(ax=taiwan_map, marker='o', color='k', markersize=5,label=f'{n_empty} unmonitored')
        gdf_lcs.plot(ax=taiwan_map, marker='o', color='orange', markersize=5,label=f'{n-n_refst-n_empty} LCSs')
        gdf_refst.plot(ax=taiwan_map, marker='o', color='#943126', markersize=5,label=f'{n_refst} Ref.St.')
        
        ax.text(119.8,25.5,f'RMSE = {RMSE:.2f} $(\mu g/m^3)$',size=text_size)
        
        ax.set_xlim(119.7,122.2)
        ax.set_ylim(21.5,26)
        
        ax.set_xlabel('Longitude (degrees)')
        ax.set_ylabel('Latitude (degrees)')
        #ax.legend(loc='upper center',ncol=3,bbox_to_anchor=(0.5, 1.15),framealpha=1)
        ax.legend(loc='upper center',ncol=1,framealpha=1,bbox_to_anchor=(0.5,1.25))
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        if save_fig:
            fname = self.save_path+f'Map_SensorPlacement_{criterion}_r{s}_N{n}_Refst_{n_refst}_Empty{n_empty}_r{s}_var{var:.2e}.png'
            fig.savefig(fname,dpi=300,format='png')
        
        return fig
        
        

#%%


if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
    # network paramteres
    N = 18
    S = 5
    YEAR = 2018
    POLLUTANT = 'O3'
    n_refst = 4
    n_empty = 8
    
    # get reduced dataset and respective locations
    dataset = load_dataset(files_path,N,POLLUTANT)
    coordinates_years = get_locations(files_path,POLLUTANT)
    coordinates = coordinates_years[YEAR][:N]
    
    # get sensors distribution from rankMax and Dopt
    var = 0e0
    if var == 0.0:
        var_Dopt = 1e-6
    else:
        var_Dopt = var
    
    df_alphas = pd.read_csv(results_path+f'Criteria_comparison/Validation_results_{N}N_{S}r.csv',index_col=0)
    alpha_reg = df_alphas.loc[n_refst,str(n_empty)]
    
    rankMax_locations, Dopt_locations, optimal_locations = get_criteria_distributions(N,S,n_refst,n_empty,alpha_reg,var,var_Dopt)
    rmse_rankMax, rmse_Dopt, rmse_optimal = get_criteria_rmse(N, S, var, var_Dopt, n_refst, n_empty)
    
    
    # map figure
    plots = Plots(save_path=results_path,marker_size=1,
                        fs_label=7,fs_ticks=7,fs_legend=5,fs_title=10,
                        show_plots=True)
    fig_rankMax = plots.geographical_network_visualization(coordinates,rankMax_locations,'rankMax',rmse_rankMax,
                                             N,n_refst,n_empty,S,var,
                                             text_size=5,save_fig=True)
    
    fig_Dopt = plots.geographical_network_visualization(coordinates,Dopt_locations,'HJB',rmse_Dopt,
                                             N,n_refst,n_empty,S,var,
                                             text_size=5,save_fig=True)
     
    fig_rankMax = plots.geographical_network_visualization(coordinates,optimal_locations,'Optimal',rmse_optimal,
                                             N,n_refst,n_empty,S,var,
                                             text_size=5,save_fig=True)
    
    
    
    
    
    
    
   