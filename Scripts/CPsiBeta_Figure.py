#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 18 11:08:22 2023

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
import matplotlib.patches as patches

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
        
    def plot_C(self,C):
        
    
        fig = plt.figure()
        ax = fig.add_subplot(111)        
        cmap = mpl.colormaps['binary'].resampled(2)
        ax.imshow(C,cmap,interpolation=None)
        for (j,i),label in np.ndenumerate(C):
            if label !=0:
                ax.text(i,j,f'{label:.0f}',ha='center',va='center',color='w',size=8)
        ax.set_xticks([])
        ax.set_xticklabels('')
        ax.set_yticks([])
        ax.set_yticklabels('')
        fname = self.save_path+'Cmatrix_figure.png'
        fig.savefig(fname,dpi=300,format='png')
    
        
    def plot_Psi(self,Psi,N,s,measurement_index):
        fig = plt.figure()
        ax = fig.add_subplot(111)        
        cmap = mpl.colormaps['YlGnBu'].resampled(N*s)
        im = ax.imshow(Psi,cmap,interpolation=None)
        for i in measurement_index:
            rect = patches.Rectangle((-1,i-0.5),width=s+1,height=1,linewidth=2,
                                     edgecolor='k',facecolor='none',
                                     clip_on=False)
            ax.add_patch(rect)
        ax.set_xticks([])
        ax.set_xticklabels('')
        ax.set_yticks([])
        ax.set_yticklabels('')
        fname = self.save_path+'Psimatrix_figure.png'
        fig.savefig(fname,dpi=300,format='png')
    
    def plot_beta(self,beta,s):
        fig = plt.figure()
        ax = fig.add_subplot(111)        
        cmap = mpl.colormaps['YlGnBu'].resampled(s)
        im = ax.imshow(beta,cmap,interpolation=None)
        ax.set_xticks([])
        ax.set_xticklabels('')
        ax.set_yticks([])
        ax.set_yticklabels('')
        fname = self.save_path+'beta_figure.png'
        fig.savefig(fname,dpi=300,format='png')
        
    def plot_y(self,y,p):
        fig = plt.figure()
        ax = fig.add_subplot(111)        
        cmap = mpl.colormaps['YlGnBu_r'].resampled(p)
        im = ax.imshow(y,cmap,interpolation=None)
        ax.set_xticks([])
        ax.set_xticklabels('')
        ax.set_yticks([])
        ax.set_yticklabels('')
        fname = self.save_path+'y_figure.png'
        fig.savefig(fname,dpi=300,format='png')
        
    
        
#%%    
if __name__=='__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
    # figure matrix size
    N = 15
    p = 5
    s = 5
    In = np.identity(N)
    rng = np.random.default_rng(seed=0)
    measurement_index = [2,5,8,11,13]#np.sort(rng.choice(np.arange(N),p,replace=False))
    C = In[measurement_index]
    Psi = rng.normal(0.0,1.0,size=(N,s))
    beta = np.arange(s)[:,None]
    y = np.array([0,2,3,2,4])[:,None]
    
    plots = Plots(save_path=results_path,marker_size=1,
                        fs_label=7,fs_ticks=7,fs_legend=5,fs_title=10,
                        show_plots=True)
    
    plots.plot_C(C)
    plots.plot_Psi(Psi,N,s,measurement_index)
    plots.plot_beta(beta,s)
    plots.plot_y(y,p)
    
    