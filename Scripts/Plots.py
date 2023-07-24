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
    def __init__(self,save_path,figx=3.5,figy=2.5,fs_label=10,fs_ticks=10,fs_legend=10,marker_size=3,dpi=300,show_plots=False):
        self.figx = figx
        self.figy = figy
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
               'dpi':self.dpi}
        ticks={'labelsize':self.fs_ticks
            }
        axes={'labelsize':self.fs_ticks,
              'grid':True
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
    
    def plot_unmonitoredNetworkPerformance(self,dicts_path,p_empty=1,metric='WCS',r=34,save_fig=False):
        sigmas = ['0.01']#['0.1','0.01','0.0001','1e-06']
        dicts_random = []
        # random placement results
        for s in sigmas :
            fname = dicts_path+f'randomPlacement_{metric}_metric_vs_p0_r{r}_sigmaZero{s}_unmonitored_pEmpty{p_empty}.pkl'
            with open(fname,'rb') as f:
                dict1 = pickle.load(f)
            dict1 = np.array([i for i in dict1.values() if type(i) is not int])
            dicts_random.append(dict1)
            
            
        dicts_method = []
        for s in sigmas:
            fname = dicts_path+f'{metric}_metric_vs_p0_r{r}_sigmaZero{s}_unmonitored_pEmpty{p_empty}.pkl'
            with open(fname,'rb') as f:
                dict_results = pickle.load(f)
            dict_results = np.array([i for i in dict_results.values()])
            # replace fails of solver output
            if np.any(dict_results==-1):
                dict_results[dict_results == -1] = np.nan
                nans, x= np.isnan(dict_results), lambda z: z.nonzero()[0]
                dict_results[nans]= np.interp(x(nans), x(~nans), dict_results[~nans])
            dicts_method.append(dict_results)
            
        dicts_2classes = []
        for s in sigmas:
            fname = dicts_path+f'{metric}_metric_vs_p0_r{r}_sigmaZero{s}_unmonitored2class_pEmpty{p_empty}.pkl'
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
        color = ['#148f77']#['#1a5276','#148f77','#d68910','#7b241c']
        label = [r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-2}$']#[r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-1}$',r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-2}$',r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-4}$',r'$\sigma_0^{2}/\sigma_{\epsilon}^{2} = 10^{-6}$']
        
        
        for p,m,t,c,l,s in zip(dicts_random,dicts_method,dicts_2classes,color,label,sigmas):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(xrange,p[:,0],color=c,label='Random placement',linestyle='--')
            ax.plot(xrange,m,color='k',label='Heuristics solution',marker='.',alpha=0.8)
            ax.plot(xrange,t,color='#a93226',label='2classes solution',marker='.',alpha=0.8)
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
            ax.legend(loc='best')
            fig.tight_layout()

        if save_fig:
            fig.savefig(self.save_path+f'{metric}_vs_num_stations_r{r}_sigmaZero{s}_unmonitored_pEmpty{p_empty}.png',dpi=600,format='png')

        
        
            
        
       
        
        
        
