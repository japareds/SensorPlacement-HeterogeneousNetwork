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
import os
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
        mpl.rc('grid',**grid)
        mpl.use(self.backend)
        
        
        
    def plot_singular_values(self,S,dataset_source,save_fig=False):
        """
        Plot singular values of low rank basis
        
        Parameters
        ----------
        S : numpy.ndarray
            singular values
        
        dataset_source : str
            specify wether dataset is real or synthetic
        
        Returns
        -------
        fig, fig1 : mpl figures
            normalized singular values and cumulative sum plots
        """
        
        # singular values
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot([i+1 for i in range(len(S))],S/max(S),'o',label='$\sigma_i$')   
        if dataset_source == 'cat':
            xrange = np.concatenate(([1],np.arange(5,len(S)+5,5)))
            yrange = np.arange(0.0,1.1,0.1)
        elif dataset_source == 'synthetic':
            xrange = np.concatenate(([1],np.arange(10,len(S)+10,10)))
            yrange = np.arange(0.6,1.2,0.2)
        ax.set_yscale('log')
        ax.set_ylabel('Normalizaed\n singular values')
        ax.set_xlabel('$\it{i}$th singular value')
        
     
            
        ax.set_xticks(xrange)
        ax.set_xticklabels(ax.get_xticks())
        #ax.set_title('Snapshots matrix singular values',fontsize=fs)
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot([i+1 for i in range(len(S))],np.cumsum(S)/np.sum(S),'o',color='orange',label='Cumulative energy')
        ax1.set_ylabel(r'Normalized $E_s$')
        
        if dataset_source == 'cat':
            xrange = np.concatenate(([1],np.arange(5,len(S)+5,5)))
            yrange = np.arange(0.0,1.1,0.1)
        elif dataset_source == 'synthetic':
            xrange = np.concatenate(([1],np.arange(10,len(S)+10,10)))
            yrange = np.arange(0.7,1.0,0.05)
            
        ax1.set_yticks(yrange)
        ax1.set_yticklabels([f'{np.round(i,decimals=2)}' for i in ax1.get_yticks()])
        ax1.set_xlabel('$\it{i}$th singular value')
        
        ax1.set_xticks(xrange)
        ax1.set_xticklabels(ax1.get_xticks())
        ax1.tick_params(axis='both', which='major')
        fig1.tight_layout()
        
        if save_fig:
            fig.savefig(self.save_path+'singularValues.png',dpi=600,format='png')
            fig1.savefig(self.save_path+'singularValues_cumsum.png',dpi=600,format='png')
            print(f'Figures saved in: {self.save_path}')
            

        
        return fig,fig1
    
    # =============================================================================
    # CONVEX OPTIMIZATION RESULTS
    # =============================================================================
    def plot_convexOpt_results(self,dicts_path_Dopt,dicts_path_rankMin,placement_metric,r,var_zero,p_empty,num_random_placements,alpha_reg,fold='Train',save_fig=False):
        """
        Plot Convex optimization relaxation of D-optimal or rank-min problems.
        For a given number of basis vector, ref.st. variance, number of unmonitored locations
        

        Parameters
        ----------
        dicts_path : str
            Path to files
        placement_metric : str
            Metric measured on the optimal location
        r : int
            Size of sparse basis.
        varZero : flaot
            reference station variance
        p_empty : int
            Number of unmonitored locations
        num_random_placements : int
            Number of random placement iterations
        save_fig : Bool, optional
            Save generated figure. The default is False.

        Returns
        -------
        fig : matplotlib figure
            Plot metric vs number of refeerence stations.

        """
        
        # load file with results
        ## random placement
        fname = dicts_path_Dopt+f'randomPlacement_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}_randomPlacements{num_random_placements}.pkl'
        with open(fname,'rb') as f:
            dict_results_random = pickle.load(f)
            
        dict_results_random = np.array([i for i in dict_results_random.values()])
        ## convex solution D-optimal
        fname = dicts_path_Dopt+f'ConvexProblemSolution_D_optimal_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}.pkl'
        with open(fname,'rb') as f:
            dict_results_convex_Dopt = pickle.load(f)
        ## discreatization D-optimal
        fname = dicts_path_Dopt+f'OptimalLocation_CovMat_D_optimal_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}.pkl'
        with open(fname,'rb') as f:
            dict_results_discrete_Dopt = pickle.load(f)
        
       
                
        
        ## convex rankMin reg
        fname = dicts_path_rankMin+f'ConvexProblemSolution_rankMin_reg_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
        with open(fname,'rb') as f:
            dict_results_convex_rankMin = pickle.load(f)
        
        fname = dicts_path_rankMin+f'OptimalLocation_CovMat_rankMin_reg_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
        with open(fname,'rb') as f:
            dict_results_discrete_rankMin = pickle.load(f)
            
       
                
        print(f'Doptimal convex results:\n{dict_results_convex_Dopt}\nRank convex results:\n{dict_results_convex_rankMin}')
        
        xrange = [i for i in dict_results_convex_rankMin.keys()][1:]#np.concatenate(([1],np.arange(5,r+5,5)))
        yrange = np.arange(0,400,100)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xrange,dict_results_random[1:,0],color='#a93226',label='Random placement',linestyle='--',marker='.')
        ax.fill_between(xrange,y1=dict_results_random[1:,1],y2=dict_results_random[1:,2],color='#a93226',alpha=0.5)
        
        if fold == 'Train':
            ax.plot(xrange,[-dict_results_convex_Dopt[i] for i in xrange],color='#1a5276',label='Convex solution D-opt',marker='.')
            ax.plot(xrange,[dict_results_convex_rankMin[i][1] for i in xrange],color='#ca6f1e',label=rf'Convex solution rank-max $\alpha$ = {alpha_reg}',marker='.')
        else:
            ax.plot(xrange,[dict_results_convex_Dopt[i] for i in xrange],color='#1a5276',label='Convex solution D-opt',marker='.')
            ax.plot(xrange,[dict_results_convex_rankMin[i] for i in xrange],color='#ca6f1e',label=rf'Convex solution rank-max $\alpha$ = {alpha_reg}',marker='.')
            
        ax.plot(xrange,[dict_results_discrete_Dopt[i] for i in xrange],color='#148f77',label='Discrete conversion D-opt',marker='.')
        ax.plot(xrange,[dict_results_discrete_rankMin[i] for i in xrange],color='orange',label='Discrete conversion rank-max',marker='.')
       
        
        ax.set_xticks(np.concatenate(([1],np.arange(10,xrange[-1]+10,10))))
        ax.set_xticklabels(ax.get_xticks())
        # ax.set_yticks(yrange)
        # ax.set_yticklabels(ax.get_yticks())
        
        ax.set_xlabel('Number of reference stations')
        ax.set_ylabel(r'$\log\det (\Sigma_{\hat{\beta}})$')
        
        #ax.set_title(f'{p_empty}% unmonitored locations\n$\epsilon^2$/$\sigma^2_m =$'r'${0:s}$'.format(scientific_notation(var_zero, 1)))
        ax.set_title(f'$\epsilon^2$/$\sigma^2_m =$'r'${0:s}$'.format(scientific_notation(var_zero, 1)))
        
        
        # if solving_algorithm == 'D_optimal':
        #     ax.set_title(f'Convex relaxation D-optimal problem\n{p_empty} unmonitored locations\n$\epsilon$/$\sigma_m$ = 10^{int(np.floor(np.log10(np.abs(varZero))))}')
        # elif solving_algorithm == 'rankMing_reg':
        #     ax.set_title(f'Convex relaxation rank-min problem\n{p_empty} unmonitored locations\n$\epsilon$/$\sigma_m$ = {varZero}')
        
        
        ax.tick_params(axis='both', which='major')
        ax.legend(loc='best',ncol=1)
        fig.tight_layout()
        
        if save_fig:
            fig.savefig(self.save_path+f'ConvexProblemSolution_{solving_algorithm}_vs_num_stations_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}.png',dpi=400,format='png')
        
        return fig
    
    def plot_previous_solutions_performance(self,dicts_path_Dopt,dicts_path_rankMin,placement_metric,r,var_zero,p_empty,num_random_placements,alpha_reg,plot_convex=False,save_fig=False):
        """
        Plot Convex optimization relaxation of D-optimal or rank-min problems
        and compare the results using locations from previous iterations in the convergence.
        
        For a given number of basis vector, ref.st. variance, number of unmonitored locations
        

        Parameters
        ----------
        dicts_path : str
            Path to files
        placement_metric : str
            Metric measured on the optimal location
        r : int
            Size of sparse basis.
        varZero : flaot
            reference station variance
        p_empty : int
            Number of unmonitored locations
        num_random_placements : int
            Number of random placement iterations
        track_convergence : Bool, optional
            Plot solution from previous step epsilon/sigma
        plot_convex : Bool, optional
            Plot convex objective function. If false, plot discrete conversion
        save_fig : Bool, optional
            Save generated figure. The default is False.

        Returns
        -------
        fig : matplotlib figure
            Plot metric vs number of refeerence stations.

        """
        
        # load results files
        ## random placement
        fname = dicts_path_Dopt+f'randomPlacement_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}_randomPlacements{num_random_placements}.pkl'
        with open(fname,'rb') as f:
            dict_results_random = pickle.load(f)
            
        dict_results_random = np.array([i for i in dict_results_random.values()])
        ## convex solution D-optimal
        fname = dicts_path_Dopt+f'ConvexProblemSolution_D_optimal_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}.pkl'
        with open(fname,'rb') as f:
            dict_results_convex_Dopt = pickle.load(f)
        ## discreatization D-optimal
        fname = dicts_path_Dopt+f'OptimalLocation_CovMat_D_optimal_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}.pkl'
        with open(fname,'rb') as f:
            dict_results_discrete_Dopt = pickle.load(f)
        
                
        
        ## convex rankMin reg
        fname = dicts_path_rankMin+f'ConvexProblemSolution_rankMin_reg_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
        with open(fname,'rb') as f:
            dict_results_convex_rankMin = pickle.load(f)
        
        fname = dicts_path_rankMin+f'OptimalLocation_CovMat_rankMin_reg_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
        with open(fname,'rb') as f:
            dict_results_discrete_rankMin = pickle.load(f)
            
     
        xrange = [i for i in dict_results_convex_rankMin.keys()][1:]#np.concatenate(([1],np.arange(5,r+5,5)))
        yrange = np.arange(0,400,100)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xrange,dict_results_random[1:,0],color='#a93226',label='Random placement',linestyle='--',marker='.')
        ax.fill_between(xrange,y1=dict_results_random[1:,1],y2=dict_results_random[1:,2],color='#a93226',alpha=0.5)
        
        # D-optimal plot
        if plot_convex:
            ax.plot(xrange,[-dict_results_convex_Dopt[i] for i in xrange],color='#148f77',label='Convex D-opt',marker='.')
        else:
            ax.plot(xrange,[dict_results_discrete_Dopt[i] for i in xrange],color='#148f77',label='Discrete conversion D-opt',marker='.')
            
        if var_zero==1e-4:
            if plot_convex:
                fname = dicts_path_Dopt+f'Convex_PreviousIteration_D_optimal_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc0.01_pEmpty{p_empty}.pkl'
                with open(fname,'rb') as f:
                    dict_results_discrete_Dopt_previous = pickle.load(f)
                ax.plot(xrange,[dict_results_discrete_Dopt_previous[i] for i in xrange],label=rf'Convex D-opt $\epsilon^2/\sigma^2_m$ = {var_zero*1e2:.1e}',marker='.')
            else:
                fname = dicts_path_Dopt+f'Discrete_PreviousIteration_D_optimal_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc0.01_pEmpty{p_empty}.pkl'
                with open(fname,'rb') as f:
                    dict_results_discrete_Dopt_previous = pickle.load(f)
                ax.plot(xrange,[dict_results_discrete_Dopt_previous[i] for i in xrange],label=rf'Discrete conversion D-opt $\epsilon^2/\sigma^2_m$ = {var_zero*1e2:.1e}',marker='.')
        elif var_zero == 1e-6:
            if plot_convex:
                fname = dicts_path_Dopt+f'Convex_PreviousIteration_D_optimal_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc0.01_pEmpty{p_empty}.pkl'
                with open(fname,'rb') as f:
                    dict_results_discrete_Dopt_previous = pickle.load(f)
                ax.plot(xrange,[dict_results_discrete_Dopt_previous[i] for i in xrange],label=rf'Convex D-opt $\epsilon^2/\sigma^2_m$ = {var_zero*1e4:.1e}',marker='.')
                
                fname = dicts_path_Dopt+f'Convex_PreviousIteration_D_optimal_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc0.0001_pEmpty{p_empty}.pkl'
                with open(fname,'rb') as f:
                    dict_results_discrete_Dopt_previous = pickle.load(f)
                ax.plot(xrange,[dict_results_discrete_Dopt_previous[i] for i in xrange],label=rf'Convex D-opt $\epsilon^2/\sigma^2_m$ = {var_zero*1e2:.1e}',marker='.')
            
            else:
                fname = dicts_path_Dopt+f'Discrete_PreviousIteration_D_optimal_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc0.01_pEmpty{p_empty}.pkl'
                with open(fname,'rb') as f:
                    dict_results_discrete_Dopt_previous = pickle.load(f)
                ax.plot(xrange,[dict_results_discrete_Dopt_previous[i] for i in xrange],label=rf'Discrete conversion D-opt $\epsilon^2/\sigma^2_m$ = {var_zero*1e4:.1e}',marker='.')
                
                fname = dicts_path_Dopt+f'Discrete_PreviousIteration_D_optimal_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc0.0001_pEmpty{p_empty}.pkl'
                with open(fname,'rb') as f:
                    dict_results_discrete_Dopt_previous = pickle.load(f)
                ax.plot(xrange,[dict_results_discrete_Dopt_previous[i] for i in xrange],label=rf'Discrete conversion D-opt $\epsilon^2/\sigma^2_m$ = {var_zero*1e2:.1e}',marker='.')
            
        # rank-max plot
        if plot_convex:
            
            ax.plot(xrange,[dict_results_convex_rankMin[i][1] for i in xrange],color='orange',label='Convex rank-max',marker='.')
        else:
            ax.plot(xrange,[dict_results_discrete_rankMin[i] for i in xrange],color='orange',label='Discrete conversion rank-max',marker='.')
            
        if var_zero ==1e-4:
            if plot_convex:
                fname = dicts_path_rankMin+f'Convex_PreviousIteration_rankMin_reg_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc0.01_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
                with open(fname,'rb') as f:
                    dict_results_discrete_rankMin_previous = pickle.load(f)
                ax.plot(xrange,[dict_results_discrete_rankMin_previous[i] for i in xrange],label=rf'Convex rank-max $\epsilon^2/\sigma^2_m$ = {var_zero*1e2:.1e}',marker='.')
            else:
                fname = dicts_path_rankMin+f'Discrete_PreviousIteration_rankMin_reg_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc0.01_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
                with open(fname,'rb') as f:
                    dict_results_discrete_rankMin_previous = pickle.load(f)
                ax.plot(xrange,[dict_results_discrete_rankMin_previous[i] for i in xrange],label=rf'Discrete conversion rank-max $\epsilon^2/\sigma^2_m$ = {var_zero*1e2:.1e}',marker='.')
        
        elif var_zero == 1e-6:
            if plot_convex:
                fname = dicts_path_rankMin+f'Convex_PreviousIteration_rankMin_reg_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc0.01_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
                with open(fname,'rb') as f:
                    dict_results_discrete_rankMin_previous = pickle.load(f)
                ax.plot(xrange,[dict_results_discrete_rankMin_previous[i] for i in xrange],label=rf'Convex rank-max $\epsilon^2/\sigma^2_m$ = {var_zero*1e4:.1e}',marker='.')
                
                fname = dicts_path_rankMin+f'Convex_PreviousIteration_rankMin_reg_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc0.0001_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
                with open(fname,'rb') as f:
                    dict_results_discrete_rankMin_previous = pickle.load(f)
                ax.plot(xrange,[dict_results_discrete_rankMin_previous[i] for i in xrange],label=rf'Convex rank-max $\epsilon^2/\sigma^2_m$ = {var_zero*1e2:.1e}',marker='.')
            
            else:
                fname = dicts_path_rankMin+f'Discrete_PreviousIteration_rankMin_reg_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc0.01_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
                with open(fname,'rb') as f:
                    dict_results_discrete_rankMin_previous = pickle.load(f)
                ax.plot(xrange,[dict_results_discrete_rankMin_previous[i] for i in xrange],label=rf'Discrete conversion rank-max $\epsilon^2/\sigma^2_m$ = {var_zero*1e4:.1e}',marker='.')
                
                fname = dicts_path_rankMin+f'Discrete_PreviousIteration_rankMin_reg_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_sigmaZeroLoc0.0001_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
                with open(fname,'rb') as f:
                    dict_results_discrete_rankMin_previous = pickle.load(f)
                ax.plot(xrange,[dict_results_discrete_rankMin_previous[i] for i in xrange],label=rf'Discrete conversion rank-max $\epsilon^2/\sigma^2_m$ = {var_zero*1e2:.1e}',marker='.')
                
        
        ax.set_xticks(np.concatenate(([1],np.arange(10,xrange[-1]+10,10))))
        ax.set_xticklabels(ax.get_xticks())
        # ax.set_yticks(yrange)
        # ax.set_yticklabels(ax.get_yticks())
        
        ax.set_xlabel('Number of reference stations')
        ax.set_ylabel(r'$\log\det (\Sigma_{\hat{\beta}})$')
        
        #ax.set_title(f'{p_empty} unmonitored locations\n$\epsilon^2$/$\sigma^2_m$ = 10^{int(np.floor(np.log10(np.abs(varZero))))}')
        #ax.set_title(f'{p_empty} unmonitored locations\n'f'$\epsilon^2$/$\sigma^2_m$ = {varZero:.1e}')
        ax.set_title(f'{p_empty}% unmonitored locations\n$\epsilon^2$/$\sigma^2_m =$'r'${0:s}$'.format(scientific_notation(var_zero, 1)))
        
        
        # if solving_algorithm == 'D_optimal':
        #     ax.set_title(f'Convex relaxation D-optimal problem\n{p_empty} unmonitored locations\n$\epsilon$/$\sigma_m$ = 10^{int(np.floor(np.log10(np.abs(varZero))))}')
        # elif solving_algorithm == 'rankMing_reg':
        #     ax.set_title(f'Convex relaxation rank-min problem\n{p_empty} unmonitored locations\n$\epsilon$/$\sigma_m$ = {varZero}')
        
        
        ax.tick_params(axis='both', which='major')
        ax.legend(loc='best',ncol=1)
        fig.tight_layout()
        
        if save_fig:
            fig.savefig(self.save_path+f'DiscreteSolution_comparison_{solving_algorithm}_vs_num_stations_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}.png',dpi=400,format='png')
        
        return fig

        
    
    def plot_fullNetworkPerformance(self,dicts_path,r=34,n=44,metric='WCS',split_figure=False,save_fig=False,alternative=False):
        """
        Plot performance of random placement and optiml configuration found by a certain algorithm 
        The performance is measured according to a specific metric.
        The performance is obtained for different values of p_zero < r and different parameter var_zero

        Parameters
        ----------
        dicts_path : str
            directory where dictionaries with results are stored
        metric : str
            scoring metric to measure location performance. D-optimal, E-optimal or WCS
        split_figure : bool
            Split figure into 4 different figures. Each showing a graph
        save_fig : bool
            save generated plot

        Returns
        -------
        fig : matplotlib figure
            plot of determinant of covariance matrix of regressor for different p_zero values

        """
       
            
        # fname = dicts_path+f'randomPlacement_{metric}_metric_vs_p0_r{r}_sigmaZero0.1.pkl'
        # with open(fname,'rb') as f:
        #     dict1 = pickle.load(f)
        
        # fname = dicts_path+f'randomPlacement_{metric}_metric_vs_p0_r{r}_sigmaZero0.01.pkl'
        # with open(fname,'rb') as f:
        #     dict2 = pickle.load(f)
        
        # fname = dicts_path+f'randomPlacement_{metric}_metric_vs_p0_r{r}_sigmaZero0.0001.pkl'
        # with open(fname,'rb') as f:
        #     dict4 = pickle.load(f)
        
        # fname = dicts_path+f'randomPlacement_{metric}_metric_vs_p0_r{r}_sigmaZero1e-06.pkl'
        # with open(fname,'rb') as f:
        #     dict6 = pickle.load(f)
        
        # dict1 = np.array([i for i in dict1.values()])
        # dict2 = np.array([i for i in dict2.values()])
        # dict4 = np.array([i for i in dict4.values()])
        # dict6 = np.array([i for i in dict6.values()])
        # dicts = [dict1,dict2,dict4,dict6]
        
        # optimal location results
        sigmas = ['0.1','0.01','0.0001','1e-06']
        dicts = []
        # random placement results
        for s in sigmas :
            if alternative:
                fname = dicts_path+f'randomPlacement_{metric}_metric_vs_p0_r{r}_sigmaZero{s}_alternativeMethod.pkl'
            else:
                fname = dicts_path+f'randomPlacement_{metric}_metric_vs_p0_r{r}_sigmaZero{s}.pkl'
            with open(fname,'rb') as f:
                dict1 = pickle.load(f)
            dict1 = np.array([i for i in dict1.values() if type(i) is not int])
            dicts.append(dict1)
            
            
        dicts_optimal = []
        for s in sigmas:
            if alternative:
                fname = dicts_path+f'{metric}_metric_vs_p0_r{r}_sigmaZero{s}_alternativeMethod.pkl'
                with open(fname,'rb') as f:
                    dict_results = pickle.load(f)
                dict_results = np.array([i for i,v in zip(dict_results.values(),dict_results.keys()) if v>=r])
            else:
                fname = dicts_path+f'{metric}_metric_vs_p0_r{r}_sigmaZero{s}.pkl'
                with open(fname,'rb') as f:
                    dict_results = pickle.load(f)
                dict_results = np.array([i for i in dict_results.values()])
            # replace fails of solver output
            if np.any(dict_results==-1):
                dict_results[dict_results == -1] = np.nan
                nans, x= np.isnan(dict_results), lambda z: z.nonzero()[0]
                dict_results[nans]= np.interp(x(nans), x(~nans), dict_results[~nans])
            dicts_optimal.append(dict_results)
                    
                
        
        
        if alternative:
            xrange = np.arange(r,n+1,1)
        else:
            xrange = np.arange(0,r+1,1)
        
        
        color = ['#1a5276','#148f77','#d68910','#7b241c']
        label = [r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-1}$',r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-2}$',r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-4}$',r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-6}$']
        names = ['sigmaZero1e-1','sigmaZero1e-2','sigmaZero1e-4','sigmaZero1e-6']
        
        if split_figure:
            for p,c,l,n,d,s in zip(dicts,color,label,names,dicts_optimal,sigmas):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(xrange,p[:,0],color=c,label='Random placement',linestyle='--')
                ax.plot(xrange,d,color='k',label=f'{metric} solution',marker='.',alpha=0.8)
                ax.fill_between(xrange,y1=p[:,1],y2=p[:,2],color=c,alpha=0.5)
                if alternative:
                    s = float(s)
                    ax.set_yscale('log')
                    ax.set_yticks(np.logspace(int(np.log10(float(s))),0,int(abs(np.log10(float(s))))+1))
                    #ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
                else:
                    ax.set_yscale('log')
                    ax.set_yticks(np.logspace(int(np.log10(float(s))),0,int(abs(np.log10(float(s))))+1))
                    # ax.set_yticks(np.arange(0.,1.2,0.2))
                    # ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
        
                ax.set_ylabel(r'$\max (\Sigma_{\hat{\beta}})_{ii}$')
                if r<=10:
                    ax.set_xticks(np.arange(10,xrange[-1]+5,5))
                else:
                    ax.set_xticks(np.arange(xrange[0],xrange[-1]+5,5))
                    
                ax.set_xticklabels([int(i) for i in ax.get_xticks()])
                ax.set_xlabel('Number of reference stations')
                ax.set_title(l)
            
                ax.tick_params(axis='both', which='major')
                ax.legend(loc='best')
                fig.tight_layout()
                if save_fig:
                    if alternative:
                        
                        fig.savefig(self.save_path+f'{metric}_vs_num_stations_{n}_r{r}_alternativeMethod.png',dpi=600,format='png')
                    else:
                        fig.savefig(self.save_path+f'{metric}_vs_num_stations_{n}_r{r}.png',dpi=600,format='png')

                
        
        else:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            
            for p,c,l in zip(dicts,color,label):
                ax.plot(xrange,p[:,0],color=c,label=l,linestyle='--')
                ax.fill_between(xrange,y1=p[:,1],y2=p[:,2],color=c,alpha=0.5)
        
                
            if metric == 'D_optimal':
                ax.set_yscale('log')
                ax.set_yticks(np.logspace(-200,0,6))
                ax.set_yticklabels([rf'$10^{ {int(np.floor(np.log10(np.abs(i))))} }$'for i in ax.get_yticks()])
                ax.set_ylabel(r'det($\Sigma_{\hat{\beta}}$)')
            elif metric == 'WCS':
                ax.set_yticks(np.arange(0.,1.2,0.2))
                ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])
                ax.set_ylabel(r'$\max (\Sigma_{\hat{\beta}})_{ii}$')
                
            ax.set_xticks(np.arange(xrange[0],xrange[-1],5))
            ax.set_xticklabels(np.arange(xrange[0],xrange[-1],5))
            
            ax.set_xlabel('Number of reference stations')
        
            ax.tick_params(axis='both', which='major')
            ax.legend(loc='lower left')
            fig.tight_layout()
            
            if save_fig:
                fig.savefig(self.save_path+f'{metric}_vs_num_stations_r{r}.png',dpi=600,format='png')
            
        
        return fig
    
    def plot_unmonitoredNetworkPerformance(self,dicts_path,p_empty=1,metric='D_optimal',r=34,sigma_ratio=['0.01'],save_fig=False):
        dicts_random = []
        # random placement results
        for s in sigma_ratio :
            fname = dicts_path+f'randomPlacement_{metric}_metric_vs_p0_r{r}_sigmaZero{s}_pEmpty{p_empty}.pkl'
            with open(fname,'rb') as f:
                dict1 = pickle.load(f)
            dict1 = np.array([i for i in dict1.values() if type(i) is not int])
            dicts_random.append(dict1)
            
            
        # rank-min solution
        dicts_method = []
        for s in sigma_ratio:
            fname = dicts_path+f'unmonitoredalgorithmPlacement_{metric}_metric_vs_p0_r{r}_sigmaZero{s}_pEmpty{p_empty}.pkl'
            
            with open(fname,'rb') as f:
                dict_results = pickle.load(f)
            dict_results = np.array([i for i in dict_results.values()])
            # replace fails of solver output
            if np.any(dict_results==-1):
                dict_results[dict_results == -1] = np.nan
                nans, x= np.isnan(dict_results), lambda z: z.nonzero()[0]
                dict_results[nans]= np.interp(x(nans), x(~nans), dict_results[~nans])
            dicts_method.append(dict_results)
        
        # two-class solution
        dicts_2classes = []
        for s in sigma_ratio:
            fname = dicts_path+f'D_optimalalgorithmPlacement_{metric}_metric_vs_p0_r{r}_sigmaZero{s}_pEmpty{p_empty}.pkl'
            with open(fname,'rb') as f:
                dict_results = pickle.load(f)
            dict_results = np.array([i for i in dict_results.values()])
            # replace fails of solver output
            if np.any(dict_results==-1):
                dict_results[dict_results == -1] = np.nan
                nans, x= np.isnan(dict_results), lambda z: z.nonzero()[0]
                dict_results[nans]= np.interp(x(nans), x(~nans), dict_results[~nans])
            dicts_2classes.append(dict_results)
            
        xrange = np.arange(0,r+1,1)
        if sigma_ratio[0] == '0.1':
            
            color = ['#1a5276']#['#1a5276','#148f77','#d68910','#7b241c']
            label = [r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-1}$']
            
        elif sigma_ratio[0] == '0.01':
            
            color = ['#148f77']#['#1a5276','#148f77','#d68910','#7b241c']
            label = [r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-2}$']
        
        elif sigma_ratio[0] == '0.001':
            
            color = ['#d68910']#['#1a5276','#148f77','#d68910','#7b241c']
            label = [r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-3}$']
        
        
        for p,m,t,c,l,s in zip(dicts_random,dicts_method,dicts_2classes,color,label,sigma_ratio):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(xrange,p[:,0],color=c,label='Random placement',linestyle='--')
            ax.plot(xrange,m,color='k',label='rank-min solution',marker='.',alpha=0.8)
            ax.plot(xrange,t,color='#a93226',label='two-class solution',marker='.',alpha=0.8)
            ax.fill_between(xrange,y1=p[:,1],y2=p[:,2],color=c,alpha=0.5)
            
            
            # ax.set_yticks(np.arange(0.,1.2,0.2))
            # ax.set_yticklabels([np.round(i,2) for i in ax.get_yticks()])

            if metric=='WCS':
                ax.set_yscale('log')
                ax.set_yticks(np.logspace(int(np.log10(float(s))),0,int(abs(np.log10(float(s))))+1))
                ax.set_ylabel(r'$\max (\Sigma_{\hat{\beta}})_{ii}$')
            elif metric=='D_optimal':
                ax.set_ylabel(r'$\log\det (\Sigma_{\hat{\beta}})$')
                ax.set_yticks(np.arange(-150,50,50))
            
            ax.set_xticks(np.arange(xrange[0],xrange[-1]+5,5))
            ax.set_xticklabels([int(i) for i in ax.get_xticks()])
            ax.set_xlabel('Number of reference stations')
            if p_empty == 1:
                ax.set_title(f'{p_empty} unmonitored location\n'+l)
            else:
                ax.set_title(f'{p_empty} unmonitored locations\n'+l)
        
            ax.tick_params(axis='both', which='major')
            ax.legend(loc='lower left',ncol=1)
            fig.tight_layout()

        if save_fig:
            fig.savefig(self.save_path+f'{metric}_vs_num_stations_r{r}_sigmaZero{s}_pEmpty{p_empty}.png',dpi=600,format='png')

    # =============================================================================
    # REGULARIZATION PLOTS
    # =============================================================================
    def plot_Validation_metric(self,dicts_path,algorithm,metric,r,VarZero,p_empty,alpha,data_source,random_placements,save_fig=False):
        
        # load results
        fname = dicts_path+f'Validation_{algorithm}_{metric}_vs_p0_r{r}_sigmaZero{VarZero}_pEmpty{p_empty}_alpha{alpha}.pkl'
        fname_random = dicts_path+f'Validation_randomPlacement_{metric}_vs_p0_r{r}_sigmaZero{VarZero}_pEmpty{p_empty}_randomPlacements{random_placements}.pkl'
        
        with open(fname,'rb') as f:
            dict_results = pickle.load(f)
        with open(fname_random,'rb') as f:
            dict_results_random = pickle.load(f)
            
        dict_results = np.array([i for i in dict_results.values()])
        dict_results_random = np.array([i for i in dict_results_random.values()])
        # replace fails of solver output
        if np.any(dict_results==-1):
            dict_results[dict_results == -1] = np.nan
            nans, x= np.isnan(dict_results), lambda z: z.nonzero()[0]
            dict_results[nans]= np.interp(x(nans), x(~nans), dict_results[~nans])
        
        xrange = np.arange(0,r+1)
        #if VarZero == 0.1:
        color = '#1a5276'#['#1a5276','#148f77','#d68910','#7b241c']
        title = rf'$\sigma_r^2/\sigma_m^2 = {VarZero}$'
    
       
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xrange,dict_results_random[:,0],color=color,label='Random placement',linestyle='--')
        ax.plot(xrange,dict_results,color='k',label='rank-min solution',marker='.',alpha=0.8)
        ax.fill_between(xrange,y1=dict_results_random[:,1],y2=dict_results_random[:,2],color=color,alpha=0.5)
        
        ax.set_ylabel(r'$\log\det (\Sigma_{\hat{\beta}})$')
        #ax.set_yticks(np.arange(-150,50,50))
        
        ax.set_xticks(np.arange(xrange[0],xrange[-1]+5,5))
        ax.set_xticklabels([int(i) for i in ax.get_xticks()])
        ax.set_xlabel('Number of reference stations')
        
        if p_empty == 1:
            ax.set_title(f'{p_empty} unmonitored location\n'+title)
        else:
            ax.set_title(f'{p_empty} unmonitored locations\n'+title)
            
        ax.tick_params(axis='both', which='major')
        ax.legend(loc='upper right',ncol=1)
        fig.tight_layout()

        if save_fig:
            print(f'Saved on {self.save_path}')
            fig.savefig(self.save_path+f'{metric}_vs_num_stations_r{r}_sigmaZero{VarZero}_pEmpty{p_empty}.png',dpi=600,format='png')

        
        return fig

    def plot_regularization_vs_refst(self,dicts_path,solving_algorithm,placement_metric,r,varZero,p_empty,save_fig=False):
        """
        Plot logDet (or other metric) vs number of reference stations for different regularization hyperparamters values.
        Two different lines: convex and discretization for each alpha_reg value

        Parameters
        ----------
        dicts_path : str
            path to files
        solving_algorithm : str
            sensor placement criteria.
        placement_metric : str
            metric computed on covariance matrix.
        r : int
            number of eigenmodes of reduced basis
        varZero : flaot
            variances ratio between ref.st. and LCSs. 
        p_empty : int 
            number of unmonitored locations
        save_fig : bool, optional
            save figure. The default is False.

        Returns
        -------
        None.

        """
        alpha_range = np.logspace(-3,3,7)
        
        # load results 
        dicts_results_alpha = {el:0 for el in alpha_range}
        dicts_results_convex_alpha = {el:0 for el in alpha_range}
        dict
        for a in alpha_range:
            # discrete locations
            fname = dicts_path+f'Validation_{solving_algorithm}_{placement_metric}_vs_p0_r{r}_sigmaZero{varZero}_pEmpty{p_empty}_alpha{a}.pkl'
            with open(fname,'rb') as f:
                dict_results_alpha = pickle.load(f)
            dicts_results_alpha[a] = np.array([i for i in dict_results_alpha.values()])
            # convex weights results
            fname = dicts_path+f'Validation_ConvexProblemSolution_{solving_algorithm}_vs_p0_r{r}_sigmaZero{varZero}_pEmpty{p_empty}_alpha{a}.pkl'
            with open(fname,'rb') as f:
                dict_results_convex_alpha = pickle.load(f)
            dicts_results_convex_alpha[a] = np.array([i for i in dict_results_convex_alpha.values()])
            
            #dict_results_alpha = np.array([i for i in dict_results_alpha.values()])
        
            # # replace fails of solver output
            # if np.any(dict_results_alpha==-1):
            #     dict_results_alpha[dict_results_alpha == -1] = np.nan
            #     nans, x= np.isnan(dict_results_alpha), lambda z: z.nonzero()[0]
            #     dict_results_alpha[nans]= np.interp(x(nans), x(~nans), dict_results_alpha[~nans])
            # dicts_results_alpha[a] = dict_results_alpha
        
        colors = ['#1a5276','#148f77','#d68910','#7b241c','#633974']
        xrange = [i for i in dict_results_alpha.keys()]#np.arange(0,int(r)+1,1)
        yrange = np.arange(-250,100,50)
        
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for a in alpha_range:
            #ax.plot(xrange,dicts_results_alpha[a],label=fr'discrete $\alpha$ = {a}',linestyle='--',marker='.')
            ax.plot(xrange,dicts_results_convex_alpha[a],label=fr'Convex $\alpha$ = {a}',linestyle='--',marker='.')
        
        ax.set_xticks(np.arange(xrange[0],xrange[-1]+10,10))
        ax.set_xticklabels([int(i) for i in ax.get_xticks()])
        ax.set_xlabel('Number of reference stations')
        
        
        
        # ax.set_yticks(yrange)
        # ax.set_yticklabels(ax.get_yticks())
        ax.set_ylabel(r'$\log\det (\Sigma_{\hat{\beta}})$')
        ax.set_title(f'{p_empty} unmonitored locations\n'rf'$\epsilon^2$/$\sigma^2_m$ = {varZero:.1e}')
        
        ax.legend(loc='upper right',ncol=2)
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        if save_fig:
            fig.savefig(self.save_path+f'Regularization_vs_numRefst_{solving_algorithm}_{placement_metric}_vs_num_stations_r{r}_sigmaZero{varZero}_pEmpty{p_empty}.png',dpi=600,format='png')


    def plot_regularization_train_val(self,train_path,val_path,solving_algorithm,placement_metric,r,p_empty,var_zero,alpha_reg,save_fig=False):
        """
        Plot train & validation sets comparison

        Parameters
        ----------
        train_path : str
            path to training set results
        val_path : str
            path to validation set results
        solving_algorithm : str
            sensor placement problem algorithm used
        placement_metric : str
            metric applied to covariance matrix
        r : int
            sparsity
        p_empty : int
            number of unmonitored locations
        var_zero : float
            ref.st variance
        alpha_reg : float
            rank-max regularization parameter
        save_fig : bool, optional
            save generated figure. The default is False.

        Returns
        -------
        None.

        """
        # load validation results
        fname = val_path+f'Validation_ConvexProblemSolution_{solving_algorithm}_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
        with open(fname,'rb') as f:
            dict_results_val = pickle.load(f)
     
        # # replace fails of solver output
        # if np.any(dict_results_val==-1):
        #     dict_results_val[dict_results_val == -1] = np.nan
        #     nans, x= np.isnan(dict_results_val), lambda z: z.nonzero()[0]
        #     dict_results_val[nans]= np.interp(x(nans), x(~nans), dict_results_val[~nans])
            
        #load training results
        fname = train_path+f'ConvexProblemSolution_{solving_algorithm}_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
        with open(fname,'rb') as f:
            dict_results_train = pickle.load(f)
        
        # replace fails of solver output
        # if np.any(dict_results_train==-1):
        #     dict_results_train[dict_results_train == -1] = np.nan
        #     nans, x= np.isnan(dict_results_train), lambda z: z.nonzero()[0]
        #     dict_results_train[nans]= np.interp(x(nans), x(~nans), dict_results_train[~nans])
        
        
        xrange = [i for i in dict_results_val.keys()]#np.arange(0,int(r)+1,1)
        yrange = np.arange(-250,100,50)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xrange,[i for i in dict_results_val.values()],label='Validation set',linestyle='--',marker='.',color='#1a5276')
        ax.plot([i for i in dict_results_train.keys()],[i[1] for i in dict_results_train.values()],label='Training set',linestyle='--',marker='.',color='#148f77')
        
        
        ax.set_xticks(np.arange(xrange[0],xrange[-1]+10,10))
        ax.set_xticklabels([int(i) for i in ax.get_xticks()])
        ax.set_xlabel('Number of reference stations')
        
        # ax.set_yticks(yrange)
        # ax.set_yticklabels(ax.get_yticks())
        ax.set_title(rf'$\alpha$ = {alpha_reg:.1e}'f'\n'rf'$\epsilon^2$/$\sigma^2_m$ = {var_zero:.1e}')
        ax.set_ylabel(r'$\log\det (\Sigma_{\hat{\beta}})$')
        
        ax.legend(loc='upper right',ncol=1)
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        if save_fig:
            fig.savefig(self.save_path+f'Regularization_train_validation_{solving_algorithm}_{placement_metric}_vs_num_stations_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}.png',dpi=600,format='png')

    
    def plot_regularization_vs_variance(self,train_path,val_path,solving_algorithm,placement_metric,r,p_empty,p_zero,alpha_range,save_fig=False):
        
        #alpha_range = [1e-3,1e-2,1e-1]#np.logspace(-1,1,3)
        variances = np.logspace(-6,-2,3)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        colors_train = ['#148f77','#ca6f1e','#a93226']
        colors_val = ['#1a5276','orange','#ec7063']
        
        for alpha_reg,ct,cv in zip(alpha_range,colors_train,colors_val):
            dict_val = {el:0 for el in variances}
            dict_train = {el:0 for el in variances}
            
            for var_zero in variances:
                # load validation results
                fname = val_path+f'Validation_ConvexProblemSolution_{solving_algorithm}_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
                with open(fname,'rb') as f:
                    dict_results_val = pickle.load(f)
                dict_val[var_zero] = dict_results_val[p_zero]
                    
                #load training results
                fname = train_path+f'ConvexProblemSolution_{solving_algorithm}_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
                with open(fname,'rb') as f:
                    dict_results_train = pickle.load(f)
                dict_train[var_zero] = dict_results_train[p_zero]

        
        
            ax.plot([i for i in range(variances.shape[0])],[dict_train[i][1] for i in dict_train.keys()],label=rf'Training set $\alpha = ${alpha_reg:.1e}',linestyle='--',marker='D',color=ct)
            ax.plot([i for i in range(variances.shape[0])],[dict_val[i] for i in dict_val.keys()],label=rf'Validation set $\alpha = ${alpha_reg:.1e}',linestyle='--',marker='.',color=cv)
            
        ax.set_xticks([i for i in range(variances.shape[0])])
        ax.set_xticklabels([f'{i:.1e}' for i in variances])
        ax.set_xlabel(r'$\epsilon^2/\sigma^2_m$')
        ax.set_ylabel(r'$\log\det (\Sigma_{\hat{\beta}})$')
        ax.set_title(f'{p_empty} unmonitored stations\n'f'{p_zero} ref.st.')
        ax.legend(loc='best',ncol=2)
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        

    def plot_validation_convex_metrics(self,dicts_path,solving_algorithm,placement_metric,r,p_empty,var_zero,alpha_reg,save_fig=False):
        """
        Plot both convex optimization metricss in the multiobjective problem:
            - log det LCSs placement
            - Trace Ref.St. placement

        Parameters
        ----------
        dicts_path : str
            path to files.
        solving_algorithm : str
            algorithm for solving
        placement_metric : str
            metric measured on covariance matrix
        r : int
            sparsity
        p_empty : int
            number of unmonitored locations
        var_zero : float
            ref.st. variance
        alpha_reg : float
            regularization parameter for trace objective function
        save_fig : bool, optional
            Save figure. The default is False.

        Returns
        -------
        None.

        """
        # load results 
        fname = dicts_path+f'Validation_ConvexObjectives_logdet&trace_{solving_algorithm}_{placement_metric}_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
        with open(fname,'rb') as f:
            dict_results_alpha = pickle.load(f)
        
        print(dict_results_alpha)
        xrange = [i for i in dict_results_alpha.keys()]#np.arange(0,int(r)+1,1)
        color1='orange'
        color2='#1a5276'
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        
        ax.plot(xrange,[dict_results_alpha[i][0] for i in dict_results_alpha.keys()],label=rf'$\alpha$ = {alpha_reg}',linestyle='--',marker='.',color=color1)
        ax2.plot(xrange,[dict_results_alpha[i][1] for i in dict_results_alpha.keys()],label='Trace',linestyle='--',marker='.',color=color2)
        
        ax.set_xticks(np.arange(xrange[0],xrange[-1]+10,10))
        ax.set_xticklabels([int(i) for i in ax.get_xticks()])
        ax.set_xlabel('Number of reference stations')
        ax.set_yticks(np.arange(-300,100,100))
        ax.set_yticklabels(ax.get_yticks())
        ax.set_ylabel(r'$\log\det \Theta^T\Theta$',color=color1)
        ax2.set_ylabel(r'$\alpha$ Tr $\Theta_r^T\Theta_r$',color=color2)
        ax2.ticklabel_format(axis='y', scilimits=[0, 0])
        
        ax.set_title(f'{p_empty} unmonitored locations\n'rf'$\epsilon^2$/$\sigma^2_m$ = {var_zero:.1e}'f'\n'rf'$\alpha$ = {alpha_reg:.1e}')
        
        #ax.legend(loc='lower left',ncol=1)
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        if save_fig:
            fig.savefig(self.save_path+f'ConvexObjective_logdet&trace_vs_numRefst_{solving_algorithm}_{placement_metric}_vs_num_stations_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}.png',dpi=600,format='png')

    
    # =============================================================================
    # PLOTS logdet
    # =============================================================================
    def plot_logdet_vs_variances(self,dict_logdet_random_var,dict_logdet_Dopt_var,dict_logdet_rankMax_var,r,n,p_zero,p_empty,save_fig=False):
        # load weights Dopt for different variances ratio
        variances = [i for i in dict_logdet_Dopt_var.keys()]
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        
        ax.plot(variances,[i for i in dict_logdet_random_var.values()],color='#a93226',label='Best random placement',linestyle='-')
        ax.plot(variances,[i for i in dict_logdet_Dopt_var.values()],color='#1a5276',label='D-optimal',linestyle='-')
        ax.plot(variances,[i for i in dict_logdet_rankMax_var.values()],color='#ca6f1e',label='Rank-max',linestyle='-')
        
        ax.set_xticks(variances)
        ax.set_xticklabels([f'{i:.1e}' for i in ax.get_xticks()])
        ax.set_xscale('log')
        
        ax.set_xlabel(r'$\epsilon^2/\sigma^2_m$')
        ax.set_ylabel(r'$\log\det\Sigma_{\hat{\beta}}$')
        
        #ax.set_yticks(np.arange(-300,100,100))
        #ax.set_yticklabels(ax.get_yticks())
            
        ax.set_title(f'{p_zero} reference stations and {n-p_empty-p_zero} LCSs')
        #ax.set_yscale('log')
        ax.legend(loc='best',ncol=1)
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        if save_fig:
            fname = f'{self.save_path}logdet_vs_variances_r{r}_pZero{p_zero}_pEmpty{p_empty}.png'
            fig.savefig(fname,dpi=300,format='png')
  
    # =============================================================================
    # PLOTS RMSE
    # =============================================================================
    
    def plot_rmse_validation(self,val_path,p_zero_plot,save_fig=False):
        
        # load results
        fname = val_path+f'RMSE_All_train_{p_zero_plot}RefSt.pkl'
        with open(fname,'rb') as f:
            rmse_train = pickle.load(f)
        
        fname = val_path+f'RMSE_All_validation_{p_zero_plot}RefSt.pkl'
        with open(fname,'rb') as f:
            rmse_val = pickle.load(f)
        
        alphas = np.logspace(-2,2,5)
      
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(alphas,[i for i in rmse_train.values()],color='#1a5276',label='Training set')
        ax.plot(alphas,[i for i in rmse_val.values()],color='orange',label='Validation set')
        ax.set_xscale('log')
        ax.set_xlabel(r'$\alpha$')
        ax.set_xticks(alphas)
        ax.set_xticklabels(ax.get_xticks())
        ax.set_ylabel('RMSE')
        
        ax.set_title(f'{p_zero_plot} reference stations')
        #ax.set_yscale('log')
        ax.legend(loc='best',ncol=1)
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        if save_fig:
            fname = f'{self.save_path}ValidationPlot_RMSE_alphas_{p_zero_plot}RefSt.png'
            fig.savefig(fname,dpi=300,format='png')
            
            
       
    def plot_other_performance_metrics(self,Dop_path,rank_path,r,var_zero,p_empty,alpha_reg,metric='RMSE',save_fig=False):
        # load results
        path = rank_path + 'RMSE_' if metric=='RMSE' else rank_path+'MAE_'
        fname = path+f'estimation_rankMin_reg_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}_alpha{alpha_reg}.pkl'
        with open(fname,'rb') as f:
            dict_rmse_rank = pickle.load(f)
            
        path = Dop_path + 'RMSE_' if metric=='RMSE' else Dop_path+'MAE_'
        fname = path+f'estimation_D_optimal_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}.pkl'
        with open(fname,'rb') as f:
            dict_rmse_Dopt = pickle.load(f)
        
        path = rank_path + 'RMSE_' if metric=='RMSE' else rank_path+'MAE_'
        fname = path+f'estimation_randomPlacement_vs_p0_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}.pkl'
        with open(fname,'rb') as f:
            dict_rmse_random = pickle.load(f)
        
        xrange = [i for i in dict_rmse_rank.keys()]
        
        print(f'rank solution\n{dict_rmse_rank}\nDopt solution\n{dict_rmse_Dopt}')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xrange[1:],[dict_rmse_random[i][0] for i in dict_rmse_random.keys()][1:],color='#a93226',label='Random placement',linestyle='-')
        ax.fill_between(xrange[1:],y1=[dict_rmse_random[i][1] for i in dict_rmse_random.keys()][1:],
                        y2=[dict_rmse_random[i][2] for i in dict_rmse_random.keys()][1:],color='#a93226',alpha=0.5)
        ax.plot(xrange[1:],[i for i in dict_rmse_rank.values()][1:],color='#ca6f1e',label='Rank-max',linestyle='-')
        ax.plot(xrange[1:],[i for i in dict_rmse_Dopt.values()][1:],color='#1a5276',label='D-optimal',linestyle='-')
        
       
        
        ax.set_xticks(np.concatenate(([xrange[1]],np.arange(10,xrange[-1]+10,10))))
        ax.set_xticklabels([int(i) for i in ax.get_xticks()])
        ax.set_xlabel('Number of reference stations')
        #ax.set_yticks(np.arange(-300,100,100))
        #ax.set_yticklabels(ax.get_yticks())
        if metric =='RMSE':
            ax.set_ylabel(r'RMSE')
        else:
            ax.set_ylabel(r'MAE')
            
        ax.set_title(f'$\epsilon^2$/$\sigma^2_m =$'r'${0:s}$'.format(scientific_notation(var_zero, 1)))
        ax.set_yscale('log')
        ax.legend(loc='upper left',ncol=1)
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        if save_fig:
            fname = self.save_path+'RMSE_' if metric=='RMSE' else self.save_path+'MAE_'
            fig.savefig(fname+f'vs_num_stations_r{r}_sigmaZero{var_zero}_pEmpty{p_empty}.png',dpi=300,format='png')

    def plot_rmse_vs_variances(self,rmse_path,p_zero,alpha_reg,locations_estimated='All',save_fig=False):
        # load results
        fname = rmse_path+f'RMSE_vs_variances_{locations_estimated}_RandomDoptRankMax_{p_zero}RefSt.pkl'
        with open(fname,'rb') as f:
            rmse_var = pickle.load(f)
        print('Loaded file\n')
        for k in rmse_var.keys():
            print(f'{k}: {rmse_var[k]}')
        
        xrange = np.arange(1,len(rmse_var)+1,1) 
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xrange,
                [i[0] for i in rmse_var.values()],color='#b03a2e',label='Best random')
        ax.plot(xrange,
                [i[1] for i in rmse_var.values()],color='#1a5276',label='D-optimal')
        ax.plot(xrange,
                [i[2] for i in rmse_var.values()],color='orange',label=rf'rankMax $\alpha=$' r'${0:s}$'.format(scientific_notation(alpha_reg, 1)))
        
        
        ax.set_xlabel(r'$\epsilon^2/\sigma^2_m$')
        ax.set_xticks(xrange)
        ax.set_xticklabels(np.concatenate(([0],[r'${0:s}$'.format(scientific_notation(i, 1)) for i in rmse_var.keys()][1:])))
            
        ax.set_yticks(np.arange(0,120,20))
        ax.set_ylim(0,100)
        ax.set_yticklabels(ax.get_yticks())
        ax.set_ylabel(r'RMSE ($\mu$g$/m^3$)')
       # ax.set_title(f'{p_zero} reference stations')
        ax.legend(loc='upper right',ncol=3)
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        if save_fig:
            fname = f'{self.save_path}Testing_RMSE_variances_{p_zero}RefSt.png'
            fig.savefig(fname,dpi=300,format='png')
            
            
        
    
        
    # =============================================================================
    # WEIGHTS AND LOCATIONS DISTRIBUTION
    # =============================================================================
    
    def plot_weights_distribution_comparison(self,dicts_path,algorithm,metric,r,varZero,p_empty,alpha,p_zero,n,save_fig=False):
        # load results 
        if algorithm == 'rankMin_reg':
            fname = dicts_path+f'Locations_{algorithm}_{metric}_vs_p0_r{r}_sigmaZero{varZero}_pEmpty{p_empty}_alpha{alpha}.pkl'
        elif algorithm == 'D_optimal':
            fname = dicts_path+f'Locations_{algorithm}_{metric}_vs_p0_r{r}_sigmaZero{varZero}_pEmpty{p_empty}.pkl'
            
        with open(fname,'rb') as f:
            dict_locs = pickle.load(f)
        
        if algorithm == 'rankMin_reg':
            fname = dicts_path+f'Weights_{algorithm}_{metric}_vs_p0_r{r}_sigmaZero{varZero}_pEmpty{p_empty}_alpha{alpha}.pkl'
        elif algorithm == 'D_optimal':
            fname = dicts_path+f'Weights_{algorithm}_{metric}_vs_p0_r{r}_sigmaZero{varZero}_pEmpty{p_empty}.pkl'
        
        with open(fname,'rb') as f:
            dict_weights = pickle.load(f)
        
        
        
        y_lcs = dict_weights[p_zero][0]
        y_refst = dict_weights[p_zero][1]
        print(y_lcs)
        print(y_refst)
        xrange = np.arange(1,n+1,1)
        yrange = np.arange(0,1.25,0.25)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.bar(xrange,y_refst,color='#1a5276',label='Ref.St.')
        ax.bar(xrange,y_lcs,color='orange',label='LCSs',bottom=y_refst)
       
        ax.set_xticks(np.concatenate(([xrange[0]],np.arange(10,n+10,10))))
        ax.set_xticklabels([i for i in ax.get_xticks()])
        ax.set_xlabel('Location index')
        
        ax.set_yticks(yrange)
        ax.set_yticklabels([np.abs(np.round(i,2)) for i in ax.get_yticks()])
        ax.set_ylim(0,1)
        ax.set_ylabel(r'$h_{r} + h$')
        
        if algorithm=='rankMin_reg':
            ax.set_title(f'{p_empty} unmonitored locations\n'rf'$\epsilon^2/\sigma^2_m = $ {var_zero:.1e}'f'\n{p_zero} reference stations selected\n'rf'$\alpha = ${alpha}')
        else:
            ax.set_title(f'{p_empty} unmonitored locations\n'rf'$\epsilon^2/\sigma^2_m = $ {var_zero:.1e}'f'\n{p_zero} reference stations selected')
        ax.legend(loc='upper right',ncol=1)
        
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        if save_fig:
            if algorithm == 'rankMin_reg':
                fname = f'WeightsDistribution_{algorithm}_{metric}_vs_p0{p_zero}_r{r}_sigmaZero{varZero}_pEmpty{p_empty}_alpha{alpha}.png'
            elif algorithm == 'D_optimal':
                fname = f'WeightsDistribution_{algorithm}_{metric}_vs_p0{p_zero}_r{r}_sigmaZero{varZero}_pEmpty{p_empty}.png'
            
            fig.savefig(self.save_path+fname,dpi=600,format='png')


    def plot_locations_rankMax_alpha(self,rank_path,r,n,p_empty,p_zero,save_fig=False):
        alphas = np.logspace(-2,2,5)
        dict_locations_alpha_refst = {f'{el:.1e}':0 for el in alphas}
        dict_locations_alpha_lcs = {f'{el:.1e}':0 for el in alphas}
        
        for alpha_reg in alphas:
            fname = rank_path+f'DiscreteLocations_rankMax_vs_p0_r{r}_pEmpty{p_empty}_alpha{alpha_reg:.1e}.pkl'
            with open(fname,'rb') as f:
                dict_loc = pickle.load(f)
                
            dict_locations_alpha_refst[f'{alpha_reg:.1e}'] = dict_loc[p_zero][1]
            
            dict_locations_alpha_lcs[f'{alpha_reg:.1e}'] = dict_loc[p_zero][0]
            
        # plot reference stations
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.vlines(x=0,ymin=0.0,ymax=n+1,colors='k')
        
        for alpha_reg,i in zip(reversed(alphas),range(alphas.shape[0])):
            ax.barh(dict_locations_alpha_refst[f'{alpha_reg:.1e}']+1,
                    1,
                    height=0.9,
                    left=i,
                    color='#1a5276'
                    )
            
            ax.vlines(x=i+1,ymin=0.0,ymax=n+1,colors='k')
        
        ax.set_yticks(np.concatenate(([1],np.arange(10,n+10,10))))
        ax.set_yticklabels(ax.get_yticks())
        
        ax.set_ylim(-1.,n+7)
        ax.set_ylabel('Location index')
        ax.set_xticks(np.arange(0.5,alphas.shape[0]+0.5,1.))
        
        ax.set_xticklabels([r'${0:s}$'.format(scientific_notation(i, 1,False)) for i in reversed(alphas)])
        
        
        ax.set_xlim(-0.1,alphas.shape[0]+0.1)
        ax.set_xlabel(r'$\alpha$')
        
        ax.set_title(f'{p_zero} reference stations selected')
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        if save_fig:
            fname = self.save_path+f'DiscreteLocations_rankMax_RefSt_vs_alpha_r{r}_pZero{p_zero}_pEmpty{p_empty}.png'
            fig.savefig(fname,dpi=300,format='png')
            
        # plot LCSs
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.vlines(x=0,ymin=0.0,ymax=n+1,colors='k')
          
        for alpha_reg,i in zip(reversed(alphas),range(alphas.shape[0])):
            ax.barh(dict_locations_alpha_lcs[f'{alpha_reg:.1e}']+1,
                    1,
                    height=0.9,
                    left=i,
                    color='#117a65'
                    )
            
            ax.vlines(x=i+1,ymin=0.0,ymax=n+1,colors='k')
        
        ax.set_yticks(np.concatenate(([1],np.arange(10,n+10,10))))
        ax.set_yticklabels(ax.get_yticks())
        
        ax.set_ylim(-1.,n+7)
        ax.set_ylabel('Location index')
        ax.set_xticks(np.arange(0.5,alphas.shape[0]+0.5,1.))
        
        ax.set_xticklabels([r'${0:s}$'.format(scientific_notation(i, 1,False)) for i in reversed(alphas)])
        
        
        ax.set_xlim(-0.1,alphas.shape[0]+0.1)
        ax.set_xlabel(r'$\alpha$')
        
        ax.set_title(f'{n-p_zero-p_empty} LCSs selected')
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        if save_fig:
            fname = self.save_path+f'DiscreteLocations_rankMax_LCS_vs_alpha_r{r}_pZero{p_zero}_pEmpty{p_empty}.png'
            fig.savefig(fname,dpi=300,format='png')
            
        # simultaneous
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.vlines(x=0,ymin=0.0,ymax=n+1,colors='k')
          
        for alpha_reg,i in zip(reversed(alphas),range(alphas.shape[0])):
            ax.barh(dict_locations_alpha_refst[f'{alpha_reg:.1e}']+1,
                    1,
                    height=0.9,
                    left=i,
                    color='#1a5276'
                    )
            ax.barh(dict_locations_alpha_lcs[f'{alpha_reg:.1e}']+1,
                    1,
                    height=0.9,
                    left=i,
                    color='#117a65'
                    )
            
            ax.vlines(x=i+1,ymin=0.0,ymax=n+1,colors='k')
        
        ax.set_yticks(np.concatenate(([1],np.arange(10,n+10,10))))
        ax.set_yticklabels(ax.get_yticks())
        
        ax.set_ylim(-1.,n+7)
        ax.set_ylabel('Location index')
        ax.set_xticks(np.arange(0.5,alphas.shape[0]+0.5,1.))
        
        ax.set_xticklabels([r'${0:s}$'.format(scientific_notation(i, 1,False)) for i in reversed(alphas)])
        
        
        ax.set_xlim(-0.1,alphas.shape[0]+0.1)
        ax.set_xlabel(r'$\alpha$')
        
        ax.set_title(f'{p_zero} reference stations and {n-p_zero-p_empty} LCSs selected')
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
       
        
        
            
    def plot_weights_evolution(self,Dopt_path,rank_path,r,n,p_empty,solving_algorithm,p_zero,alpha_reg):
        # load weights Dopt for different variances ratio
        variances = np.array([1e-6,1e-4,1e-2,1e-1,1e0])
        dict_weights_var_refst = {el:0 for el in variances}
        dict_weights_var_lcs = {el:0 for el in variances}
        
        for var in variances:
            fname = Dopt_path+f'Weights_D_optimal_vs_p0_r{r}_pEmpty{p_empty}_varZero{var:.1e}.pkl'
            with open(fname,'rb') as f:
                dict_weights = pickle.load(f)
        
            dict_weights_var_refst[var] = dict_weights[p_zero][1]
            dict_weights_var_lcs[var] = dict_weights[p_zero][0]
            
        fname = rank_path+f'Weights_rankMax_vs_p0_r{r}_pEmpty{p_empty}_alpha{alpha_reg:.1e}.pkl'
        with open(fname,'rb') as f:
            dict_weights = pickle.load(f)
        dict_weights_refst_limit = dict_weights[p_zero][1]
        dict_weights_lcs_limit = dict_weights[p_zero][0]
        
        # plot reference stations
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.vlines(x=0,ymin=0.0,ymax=dict_weights_var_refst[var].shape[0]+1,colors='k')
        colors = ['#2e86c1','#239b56','#117864','#1a5276','#4d5656']
    
        for var,i,c in zip(reversed(variances),range(variances.shape[0]),colors):
            ax.barh(np.arange(1,dict_weights_var_refst[var].shape[0]+1,1),
                    dict_weights_var_refst[var],
                    height=0.9,
                    left=i,
                    label=r'$\epsilon^2/\sigma^2_m=$'r'${0:s}$'.format(scientific_notation(var, 1,False)),
                    color=c
                    )
            ax.vlines(x=i+1,ymin=0.0,ymax=dict_weights_var_refst[var].shape[0]+1,colors='k')
        
        ax.barh(np.arange(1,dict_weights_refst_limit.shape[0]+1,1),
                dict_weights_refst_limit,
                height=0.9,
                left=i+1,
                label='rankMax',
                color='#633974')
        
        ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.0),ncol=variances.shape[0]+1)
        
        ax.set_yticks(np.concatenate(([1],np.arange(10,dict_weights_var_refst[var].shape[0]+10,10))))
        ax.set_yticklabels(ax.get_yticks())
        ax.set_ylim(-1.,dict_weights_var_refst[var].shape[0]+10)
        ax.set_ylabel('Location index')
        
        
        ax.set_xticks(np.arange(0.0,variances.shape[0]+1+0.5,0.5))
        ax.set_xticklabels(ax.get_xticks())
        ax.set_xlim(-0.1,variances.shape[0]+1+0.1)
        ax.set_xlabel(r'$h_r$')
        
        ax.set_title(f'Weights convergence\n {p_zero} reference stations')
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        # plot LCSs
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.vlines(x=0,ymin=0.0,ymax=dict_weights_var_refst[var].shape[0]+1,colors='k')
        
        for var,i,c in zip(reversed(variances),range(variances.shape[0]),colors):
            ax.barh(np.arange(1,dict_weights_var_lcs[var].shape[0]+1,1),
                    dict_weights_var_lcs[var],
                    height=0.9,
                    left=i,
                    label=r'$\epsilon^2/\sigma^2_m=$'r'${0:s}$'.format(scientific_notation(var, 1,False)),
                    color=c
                    )
            ax.vlines(x=i+1,ymin=0.0,ymax=dict_weights_var_lcs[var].shape[0]+1,colors='k')
        
        ax.barh(np.arange(1,dict_weights_lcs_limit.shape[0]+1,1),
                dict_weights_lcs_limit,
                height=0.9,
                left=i+1,
                label='rankMax',
                color='#633974')
        
        ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.0),ncol=variances.shape[0]+1)
        
        ax.set_yticks(np.concatenate(([1],np.arange(10,dict_weights_var_lcs[var].shape[0]+10,10))))
        ax.set_yticklabels(ax.get_yticks())
        ax.set_ylim(-1.,dict_weights_var_lcs[var].shape[0]+10)
        ax.set_ylabel('Location index')
        
        
        ax.set_xticks(np.arange(0.0,variances.shape[0]+1+0.5,0.5))
        ax.set_xticklabels(ax.get_xticks())
        ax.set_xlim(-0.1,variances.shape[0]+1+0.1)
        ax.set_xlabel(r'$h_r$')
        
        ax.set_title(f'Weights convergence\n {n-p_empty-p_zero} LCSs')
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        
    def plot_locations_evolution(self,Dopt_path,rank_path,r,n,p_empty,p_zero,alpha_reg,save_fig=False):
        # load weights Dopt for different variances ratio
        variances = np.array([1e-6,1e-4,1e-2,1e-1,1e0])
        dict_locations_var_refst = {el:0 for el in variances}
        dict_locations_var_lcs = {el:0 for el in variances}
        
        for var in variances:
            fname = Dopt_path+f'DiscreteLocations_D_optimal_vs_p0_r{r}_pEmpty{p_empty}_varZero{var:.1e}.pkl'
            with open(fname,'rb') as f:
                dict_locations = pickle.load(f)
            print(var)
            if dict_locations[p_zero][2].shape[0] == p_empty:
                dict_locations_var_refst[var] = np.array(dict_locations[p_zero][1])
                dict_locations_var_lcs[var] = np.array(dict_locations[p_zero][0])
            else:
                dict_locations_var_refst[var] = []
                dict_locations_var_lcs[var] = []
            print(dict_locations_var_refst[var])
            
        fname = rank_path+f'DiscreteLocations_rankMax_vs_p0_r{r}_pEmpty{p_empty}_alpha{alpha_reg:.1e}.pkl'
        with open(fname,'rb') as f:
            dict_locations = pickle.load(f)
        dict_locations_refst_limit = dict_locations[p_zero][1]
        dict_locations_lcs_limit = dict_locations[p_zero][0]
       
        
        # plot reference stations
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.vlines(x=0,ymin=0.0,ymax=n+1,colors='k')
          
        for var,i in zip(reversed(variances),range(variances.shape[0])):
            if len(dict_locations_var_refst[var]) ==0:
                ax.barh(dict_locations_var_refst[var],
                        1,
                        height=0.9,
                        left=i,
                        color='#1a5276'
                        )
                    
            else:
                ax.barh(dict_locations_var_refst[var]+1,
                        1,
                        height=0.9,
                        left=i,
                        color='#1a5276'
                        )
                
            ax.vlines(x=i+1,ymin=0.0,ymax=n+1,colors='k')
        
        ax.barh(dict_locations_refst_limit+1,
                 1,
                 height=0.9,
                 left=i+1,
                 color='#1a5276'
                 )
   
        
        ax.set_yticks(np.concatenate(([1],np.arange(10,n+10,10))))
        ax.set_yticklabels(ax.get_yticks())
        
        ax.set_ylim(-1.,n+7)
        ax.set_ylabel('Location index')
        ax.set_xticks(np.arange(0.5,variances.shape[0]+1+0.5,0.5))
        
        ax.set_xticks(np.arange(0.5,variances.shape[0]+1+0.5,1.))
      
        ax.set_xticklabels(np.concatenate(([r'${0:s}$'.format(scientific_notation(i, 1,False)) for i in reversed(variances)],['rankMax'])))
       
        ax.set_xlim(-0.1,variances.shape[0]+1+0.1)
        ax.set_xlabel(r'$\epsilon^2/\sigma^2_m$')
        
        ax.set_title(f'{p_zero} reference stations selected')
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        if save_fig:
            fname = self.save_path+f'DiscreteLocationsEvolution_RefSt_vs_variances_r{r}_pZero{p_zero}_pEmpty{p_empty}.png'
            fig.savefig(fname,dpi=300,format='png')
        
        # plot LCSs
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.vlines(x=0,ymin=0.0,ymax=n+1,colors='k')
          
        for var,i in zip(reversed(variances),range(variances.shape[0])):
            if len(dict_locations_var_lcs[var]) == 0:
                ax.barh(dict_locations_var_lcs[var],
                        1,
                        height=0.9,
                        left=i,
                        color='#117a65'
                        )
            else:
                    
            
                ax.barh(dict_locations_var_lcs[var]+1,
                        1,
                        height=0.9,
                        left=i,
                        color='#117a65'
                        )
                        
            ax.vlines(x=i+1,ymin=0.0,ymax=n+1,colors='k')
        
        ax.barh(dict_locations_lcs_limit+1,
                 1,
                 height=0.9,
                 left=i+1,
                 color='#117a65'
                 )
        # ax.barh(dict_locations_lcs_limit+1,
        #          1,
        #          height=0.9,
        #          left=i+1,
        #          label='LCSs',
        #          color='orange'
        #          )
        
        #ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.0),ncol=variances.shape[0]+1)
        
        ax.set_yticks(np.concatenate(([1],np.arange(10,n+10,10))))
        ax.set_yticklabels(ax.get_yticks())
        
        ax.set_ylim(-1.,n+7)
        ax.set_ylabel('Location index')
        ax.set_xticks(np.arange(0.5,variances.shape[0]+1+0.5,0.5))
        
        ax.set_xticks(np.arange(0.5,variances.shape[0]+1+0.5,1.))
        #ax.set_xticklabels(np.concatenate(([f'{i:.1e}' for i in reversed(variances)],['rankMax'])))
        ax.set_xticklabels(np.concatenate(([r'${0:s}$'.format(scientific_notation(i, 1,False)) for i in reversed(variances)],['rankMax'])))
        #ax.set_title(f'$\epsilon^2$/$\sigma^2_m =$'r'${0:s}$'.format(scientific_notation(var_zero, 1)))
        
        ax.set_xlim(-0.1,variances.shape[0]+1+0.1)
        ax.set_xlabel(r'$\epsilon^2/\sigma^2_m$')
        
        ax.set_title(f'{n-p_empty-p_zero} LCSs selected')
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        
        if save_fig:
            fname = self.save_path+f'DiscreteLocationsEvolution_LCS_vs_variances_r{r}_pZero{p_zero}_pEmpty{p_empty}.png'
            fig.savefig(fname,dpi=300,format='png')
            
            
        # simultaneous
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.vlines(x=0,ymin=0.0,ymax=n+1,colors='k')
          
        for var,i in zip(reversed(variances),range(variances.shape[0])):
            if len(dict_locations_var_lcs[var]) == 0:
                ax.barh(dict_locations_var_lcs[var],
                        1,
                        height=0.9,
                        left=i,
                        color='#117a65'
                        )
                ax.barh(dict_locations_var_refst[var],
                        1,
                        height=0.9,
                        left=i,
                        color='#1a5276'
                        )
                
            else:
                ax.barh(dict_locations_var_lcs[var]+1,
                        1,
                        height=0.9,
                        left=i,
                        color='#117a65'
                        )
                ax.barh(dict_locations_var_refst[var]+1,
                        1,
                        height=0.9,
                        left=i,
                        color='#1a5276'
                        )
            
                    
            ax.vlines(x=i+1,ymin=0.0,ymax=n+1,colors='k')
        
        ax.barh(dict_locations_lcs_limit+1,
                 1,
                 height=0.9,
                 left=i+1,
                 color='#117a65',
                 label = 'LCSs'
                 )
        
        ax.barh(dict_locations_refst_limit+1,
                  1,
                  height=0.9,
                  left=i+1,
                  label='Ref. St.',
                  color='#1a5276'
                  )
        
        ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.0),ncol=variances.shape[0]+1)
        
        ax.set_yticks(np.concatenate(([1],np.arange(10,n+10,10))))
        ax.set_yticklabels(ax.get_yticks())
        
        ax.set_ylim(-1.,n+7)
        ax.set_ylabel('Location index')
        ax.set_xticks(np.arange(0.5,variances.shape[0]+1+0.5,0.5))
        
        ax.set_xticks(np.arange(0.5,variances.shape[0]+1+0.5,1.))
        #ax.set_xticklabels(np.concatenate(([f'{i:.1e}' for i in reversed(variances)],['rankMax'])))
        ax.set_xticklabels(np.concatenate(([r'${0:s}$'.format(scientific_notation(i, 1,False)) for i in reversed(variances)],['rankMax'])))
        #ax.set_title(f'$\epsilon^2$/$\sigma^2_m =$'r'${0:s}$'.format(scientific_notation(var_zero, 1)))
        
        ax.set_xlim(-0.1,variances.shape[0]+1+0.1)
        ax.set_xlabel(r'$\epsilon^2/\sigma^2_m$')
        
        ax.set_title(f'{p_zero} Ref.St. and {n-p_empty-p_zero} LCSs selected')
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        

    def plot_weights_comparison_3d(self,dicts_path,algorithm,metric,r=54,varZero=0.01,p_empty=40,p_zero=40,n=100,save_fig=False):
        
        alpha_range = np.logspace(-1,1,3)
        
        # load results 
        dicts_locations_alpha = {el:0 for el in alpha_range}
        dicts_weights_alpha = {el:0 for el in alpha_range}
        
        for alpha in alpha_range:
            # load optimal locations
            if algorithm == 'rankMin_reg':
                fname = dicts_path+f'Locations_{algorithm}_{metric}_vs_p0_r{r}_sigmaZero{varZero}_pEmpty{p_empty}_alpha{alpha}.pkl'
            elif algorithm == 'D_optimal':
                fname = dicts_path+f'Locations_{algorithm}_{metric}_vs_p0_r{r}_sigmaZero{varZero}_pEmpty{p_empty}.pkl'
            
            with open(fname,'rb') as f:
                dicts_locations_alpha[alpha] = pickle.load(f)
        
            # load weights
            if algorithm == 'rankMin_reg':
                fname = dicts_path+f'Weights_{algorithm}_{metric}_vs_p0_r{r}_sigmaZero{varZero}_pEmpty{p_empty}_alpha{alpha}.pkl'
            elif algorithm == 'D_optimal':
                fname = dicts_path+f'Weights_{algorithm}_{metric}_vs_p0_r{r}_sigmaZero{varZero}_pEmpty{p_empty}.pkl'
            
            with open(fname,'rb') as f:
                dicts_weights_alpha[alpha] = pickle.load(f)
        
        
            
        xrange = np.arange(1,n+1,1)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        depth_ticks = np.arange(0,alpha_range.shape[0]+1,1)
        
        for i in range(len(alpha_range)):
            alpha = alpha_range[i]
            y_lcs = dicts_weights_alpha[alpha][p_zero][0]
            y_refst = dicts_weights_alpha[alpha][p_zero][1]
            ax.bar(xrange, y_refst, zs=i,zdir='y',color='#1a5276', alpha=0.8,width=1.2)
            
        
        ax.set_xlabel('Reference station index')
        ax.set_ylabel(r'$\alpha$')
        ax.set_zlabel('Weight')
        ax.set_xticks(np.concatenate(([1],np.arange(20,120,20))))
        ax.set_xticklabels(ax.get_xticks(),rotation=0)
        ax.set_yticks([i for i in range(alpha_range.shape[0])])
        ax.set_yticklabels(alpha_range)
        ax.set_zticks([np.round(i,1) for i in np.arange(0,1.2,0.2)])
        ax.set_zticklabels(ax.get_zticks())
        
        ax.set_title(f'{p_empty} unmonitored locations\n'rf'$\epsilon^2/\sigma^2_m$ = {varZero}'f'\n{p_zero} reference stations used')
        fig.tight_layout()
        
        
    # =============================================================================
    #         Execution time
    # =============================================================================
    def plot_execution_time_variance(self,Dopt_path,rank_path,r,p_empty,p_zero,alpha_reg,save_fig=False):
        # execution time
        variances = np.logspace(-6,0,7)
        exec_time_Dopt = {el:0 for el in variances}
        for var in variances:
            fname = Dopt_path+f'ExecutionTime_D_optimal_vs_p0_r{r}_pEmpty{p_empty}_varZero{var:.1e}.pkl'
            with open(fname,'rb') as f:
                exec_time = pickle.load(f)
            exec_time_Dopt[var] = exec_time[p_zero]
        
        
        alphas = np.logspace(-2,2,5)    
        exec_time_rank = {el:0 for el in alphas}
        for alpha_reg in alphas:
            fname = rank_path+f'ExecutionTime_rankMax_vs_p0_r{r}_pEmpty{p_empty}_alpha{alpha_reg:.1e}.pkl'
            with open(fname,'rb') as f:
                exec_time = pickle.load(f)
            exec_time_rank[alpha_reg] = exec_time[p_zero]
            
        
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(variances,[i for i in exec_time_Dopt.values()],color='#1a5276',label='D-optimal')
        ax.hlines(y=exec_time_rank[alpha_reg],xmin=variances[0],xmax=variances[-1],color='orange',label=r'rankMax $\alpha$='r'${0:s}$'.format(scientific_notation(alpha_reg, 1)))
        ax.set_xscale('log')
        ax.set_xticks(variances)
        ax.set_xticklabels(ax.get_xticks())
        ax.set_xlabel(r'$\epsilon^2/\sigma^2_m$')
        ax.set_ylabel('Execution time (s)')
        
        ax.legend(loc='upper right',ncol=1)
        ax.tick_params(axis='both', which='major')
        fig.tight_layout()
        
        if save_fig:
            fname = self.save_path+f'ExecutionTime_Dopt_rankMax_{r}_pEmpty{p_empty}_{p_zero}RefSt.png'
            fig.savefig(fname,dpi=300,format='png')

        
if __name__ == '__main__':
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    print('Loading data set')
    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    dataset_source = 'synthetic' #['synthetic','real']
    
    # network parameters
    n = 100 if dataset_source == 'synthetic' else 44
    r = 54 if dataset_source == 'synthetic' else 34
    
    num_random_placements = 100
    var_eps,var_zero = 1,1e-6
    p_empty = 40
    alpha_reg = 1e-3
    solving_algorithm = 'rankMin_reg' #['rankMin_reg','D_optimal']z
    placement_metric = 'logdet' # compute logdet on cov matrix
    
    
    plots = Plots(save_path=results_path,marker_size=1,fs_label=4,fs_legend=4,fs_title=10,show_plots=True)
    
    if dataset_source == 'real':
        solutions_path = results_path+'Unmonitored_locations/Training_Testing_split/'
    else:
        solutions_path = results_path+'Unmonitored_locations/Synthetic_Data/'
        
 
    # plot heterogeneous network with unmonitored locations
    # fig = plots.plot_Validation_metric(f'{solutions_path}ValidationSet_results/',solving_algorithm,placement_metric,r,var_zero,p_empty,alpha_reg,dataset_source,num_random_placements,save_g=False)
    # plots.plot_weightsDistribution(f'{solutions_path}TrainingSet_results/',solving_algorithm,placement_metric,r,varZero=var_zero,p_empty=p_empty,alpha=alpha_reg,p_zero=10,n=n,save_fig=False)
    
    plt.close('all')
    p_zero = 30
    plots.plot_convexOpt_results(f'{solutions_path}TrainingSet_results/D_optimal/',f'{solutions_path}TrainingSet_results/rankMin_reg/', placement_metric, r, var_zero, p_empty, num_random_placements, alpha_reg,save_fig=False)
    #plots.plot_weights_comparison_3d(f'{solutions_path}TrainingSet_results/{solving_algorithm}/',solving_algorithm,placement_metric,r,var_zero,p_empty,p_zero,n,save_fig=False)
    # plots.plot_weights_distribution_comparison(f'{solutions_path}TrainingSet_results/{solving_algorithm}/',solving_algorithm,placement_metric,r,var_zero,p_empty,alpha_reg,p_zero,n=n,save_fig=False)    
    plots.plot_weights_distribution_comparison(f'{solutions_path}TrainingSet_results/D_optimal/','D_optimal',placement_metric,r,var_zero,p_empty,alpha_reg,p_zero,n=n,save_fig=False)    
        
