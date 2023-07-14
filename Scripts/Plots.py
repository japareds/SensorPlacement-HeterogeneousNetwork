#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:58:29 2023

@author: jparedes
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
# =============================================================================
# Plots
# =============================================================================
class Plots():
    def __init__(self,save_path,figx=3.5,figy=2.5,fs_label=10,fs_ticks=10,fs_legend=10,dpi=300,show_plots=False):
        self.figx = figx
        self.figy = figy
        self.fs_label = fs_label
        self.fs_ticks = fs_ticks
        self.fs_legend = fs_legend
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
        fig = {'figsize':[self.figx,self.figy],
               'dpi':self.dpi}
        ticks={'labelsize':self.fs_ticks
            }
        axes={'labelsize':self.fs_ticks,
              'grid':True
            }
        mathtext={'default':'regular'}
        legend = {'fontsize':self.fs_legend}
        
        
        mpl.rc('font',**font)
        mpl.rc('figure',**fig)
        mpl.rc('xtick',**ticks)
        mpl.rc('ytick',**ticks)
        mpl.rc('axes',**axes)
        mpl.rc('legend',**legend)
        mpl.rc('mathtext',**mathtext)
        mpl.use(self.backend)
        
        
        
    def plot_singular_values(self,S,save_fig=False):
        """
        Plot singular values of low rank basis
        
        Parameters
        ----------
        S : numpy.ndarray
            singular values
        
        Returns
        -------
        fig, fig1 : mpl figures
            normalized singular values and cumulative sum plots
        """
        
        # singular values
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([i+1 for i in range(len(S))],S/max(S),'o',label='$\sigma_i$')    
        #ax.set_yscale('log')
        yrange = np.logspace(-2,0,3)
        ax.set_yticks(yrange)
        ax.set_yticklabels(['$10^{-2}$','$10^{-1}$','$1$'])
        ax.set_ylabel('Normalizaed singular values')
        ax.set_xlabel('$i$th singular value')
        yrange = np.arange(0.0,1.1,0.1)
        xrange = np.arange(0,len(S)+5,5)
        ax.set_xticks(xrange[1:])
        ax.set_xticklabels(ax.get_xticks())
        
        #ax.set_title('Snapshots matrix singular values',fontsize=fs)
        
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot([i+1 for i in range(len(S))],np.cumsum(S)/np.sum(S),'o',color='orange',label='Cumulative energy')
        ax1.set_ylabel('Cumulative sum')
        yrange = np.arange(0.0,1.2,0.25)
        ax1.set_yticks(yrange)
        ax1.set_yticklabels(np.round(ax1.get_yticks(),decimals=1))
        ax1.set_xlabel('$i$th singular value')
        xrange = np.arange(0,len(S)+5,5)
        ax1.set_xticks(xrange[1:])
        ax1.set_xticklabels(ax1.get_xticks())

        

        ax1.tick_params(axis='both', which='major')
        fig1.tight_layout()
        
        if save_fig:
            fig.savefig(self.save_path+'singularValues.png',dpi=600,format='png')
            fig1.savefig(self.save_path+'singularValues_cumsum.png',dpi=600,format='png')
            

        
        return fig,fig1
    
    
    def plot_D_optimal_metrics(self,dicts_path,save_fig=False):
        """
        Plot D_optimal metric (determinant of covariance matrix)
        for different values of p_zero < r and different parameter var_zero

        Parameters
        ----------
        dicts_path : str
            directory where dictionaries with results are stored

        Returns
        -------
        fig : matplotlib figure
            plot of determinant of covariance matrix of regressor for different p_zero values

        """
        
        fname = dicts_path+'D_optimal_metric_vs_p0_r34_sigmaZero0.1.pkl'
        with open(fname,'rb') as f:
            dict1 = pickle.load(f)
        
        fname = dicts_path+'D_optimal_metric_vs_p0_r34_sigmaZero0.01.pkl'
        with open(fname,'rb') as f:
            dict2 = pickle.load(f)
        
        fname = dicts_path+'D_optimal_metric_vs_p0_r34_sigmaZero0.0001.pkl'
        with open(fname,'rb') as f:
            dict4 = pickle.load(f)
        
        fname = dicts_path+'D_optimal_metric_vs_p0_r34_sigmaZero1e-06.pkl'
        with open(fname,'rb') as f:
            dict6 = pickle.load(f)
        r=34
        dict1 = np.array([[np.exp(i[0]),np.exp(i[0])-np.exp(i[1]),np.exp(i[2])-np.exp(i[0])] for i in dict1.values()])
        dict2 = np.array([[np.exp(i[0]),np.exp(i[0])-np.exp(i[1]),np.exp(i[2])-np.exp(i[0])] for i in dict2.values()])
        dict4 = np.array([[np.exp(i[0]),np.exp(i[0])-np.exp(i[1]),np.exp(i[2])-np.exp(i[0])] for i in dict4.values()])
        dict6 = np.array([[np.exp(i[0]),np.exp(i[0])-np.exp(i[1]),np.exp(i[2])-np.exp(i[0])] for i in dict6.values()])
        xrange = np.arange(0,r+1,1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        color = ['#1a5276','#148f77','#d68910','#7b241c']
        label = [r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-1}$',r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-2}$',r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-4}$',r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-6}$']
        for p,c,l in zip([dict1,dict2,dict4,dict6],color,label):
            ax.errorbar(xrange,p[:,0],yerr=p[:,1:].T,color=c,label=l,ecolor='k')
        ax.set_yscale('log')
        ax.set_xticks(np.arange(xrange[0],xrange[-1],5))
        ax.set_xticklabels(np.arange(xrange[0],xrange[-1],5))
        ax.set_yticks(np.logspace(-200,0,6))
        #ax.set_yticklabels([f'{i:.1e}'for i in ax.get_yticks()])
        ax.set_yticklabels([rf'$10^{ {int(np.floor(np.log10(np.abs(i))))} }$'for i in ax.get_yticks()])
        ax.set_xlabel('Number of reference stations')
        ax.set_ylabel(r'det($\Sigma_{\hat{\beta}}$)')
        ax.tick_params(axis='both', which='major')
        ax.legend(loc='lower left')
        fig.tight_layout()
        if save_fig:
            fig.savefig(self.save_path+'detSigma_vs_num_stations.png',dpi=600,format='png')
        
        
        return fig
        
