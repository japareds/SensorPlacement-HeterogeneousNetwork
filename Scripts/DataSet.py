#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:06:07 2023

@author: jparedes
"""
import os
import pandas as pd
import numpy as np
import pickle
# =============================================================================
# Network data set
# =============================================================================
class DataSet():
    def __init__(self,pollutant,start_date,end_date,files_path,source='real'):
        """
        Network reference stations. Each station has an assigned ID

        Parameters
        ----------
        pollutant : str
            pollutant measured by reference stations
        start_date : str
            starting date of measurements in 'YYYY-MM-DD'
        end_date : str
            end date of measurements in 'YYYY-MM-DD'
        files_path: str
            path to files
        source: str
            Use real or synthetic dataset
            
        Returns
        -------
        None.

        """
        self.stations_list = {
                       'Badalona':'08015021',
                       'Eixample':'08019043',
                       'Gracia':'08019044',
                       'Ciutadella':'08019050',
                       'Vall-Hebron':'08019054',
                       'Palau-Reial':'08019057',
                       'Fabra':'08019058',
                       'Berga':'08022006',
                       'Gava':'08089005',
                       'Granollers':'08096014',
                       'Igualada':'08102005',
                       'Manlleu':'08112003',
                       'Manresa':'08113007',
                       'Mataro':'08121013',
                       'Montcada':'08125002',
                       #'Montseny':'08125002', # linearly dependent to other measurements
                       'El-Prat':'08169009',
                       'Rubi':'08184006',
                       'Sabadell':'08187012',
                       'Sant-Adria':'08194008',
                       'Sant-Celoni':'08202001',
                       'Sant-Cugat':'08205002',
                       'Santa-Maria':'08259002',
                       'Sant-Vicenç':'08263001',
                       'Terrassa':'08279011',
                       'Tona':'08283004',
                       'Vic':'08298008',
                       'Viladecans':'08301004',
                       'Vilafranca':'08305006',
                       'Vilanova':'08307012',
                       'Agullana':'17001002',
                       'Begur':'17013001',
                       'Pardines':'17125001',
                       'Santa-Pau':'17184001',
                       'Bellver':'25051001',
                       'Juneda':'25119002',
                       'Lleida':'25120001',
                       'Ponts':'25172001',
                       'Montsec':'25196001',
                       'Sort':'25209001',
                       'Alcover':'43005002',
                       'Amposta':'43014001',
                       'La-Senla':'43044003',
                       'Constanti':'43047001',
                       'Gandesa':'43064001',
                       'Els-Guiamets':'43070001',
                       'Reus':'43123005',
                       'Tarragona':'43148028',
                       'Vilaseca':'43171002'
                       }
        
        self.RefStations = [i for i in self.stations_list.keys()]
        self.pollutant = pollutant # pollutant to load
        self.startDate = start_date 
        self.endDate = end_date 
        self.ds = pd.DataFrame()
        
        self.source = source
        
        if source == 'real':
            self.files_path = files_path
        else:
            self.files_path = files_path+'SyntheticDataSet/'
        
    def load_dataSet(self,n_years=12,n_stations=100):
        """
        Load csv files containing reference stations measurements for specified period of time

        Returns
        -------
        self.ds: pandas dataframe
            dataframe containing measurements: [num_dates x num_stations]

        """
        # load real dataset
        if self.source == 'real':
            for rs in self.RefStations:
                fname = f'{self.files_path}{self.pollutant}_{rs}_{self.startDate}_{self.endDate}.csv'
                print(f'Loading data set {fname}')
                df_ = pd.read_csv(fname,index_col=0,sep=';')
                df_.index = pd.to_datetime(df_.index)
                self.ds = pd.concat([self.ds,df_],axis=1)
            self.ds = self.ds.drop_duplicates(keep='first')
        
        # load synthetic dataset
        elif self.source == 'synthetic':
            fname = f'{self.files_path}{self.pollutant}_SyntheticData_{self.startDate}_{self.endDate}_{n_stations}stations.pkl'
            with open(fname,'rb') as f:
                self.ds = pickle.load(f)

         
            
            
            
        print(f'All data sets loaded\n{self.ds.shape[0]} measurements for {self.ds.shape[1]} reference stations')
            
            
    def train_test_split(self,train_set_end,val_set_begin,val_set_end,test_set_begin):
        """
        Split dataset into training, validation, and testing set.
        The end of training set, the start and end of validation and the end of testing set must be specified
        Dates format: 'yyyy-mm-dd hh:mm:ss'
        
        Parameters
        ----------
        train_set_end : str
                        date of last training set measurement
        val_set_begin : str
                        date of first validation set measurement
        val_set_end : str
                        date of last validation set measurement
        test_set_begin : str
                        date of first test set measurement

        Returns
        -------
        self.ds_train : pandas dataframe
                        Training set dataset
        self.ds_val : pandas dataframe
                        Validation set dataset
        self.ds_test : pandas dataframe
                        Testing set dataset

        """
        self.ds_train = self.ds.loc[:train_set_end].copy()
        self.ds_val = self.ds.loc[val_set_begin:val_set_end].copy()
        self.ds_test = self.ds.loc[test_set_begin:].copy()
        
  
        
        
    def cleanMissingvalues(self,strategy='remove',tol=0.1):
        """
        Remove missing values from data set.
        Three possibilities: 
            1) remove stations with not enough measurements
            2) drop missing values for all stations
            3) interpolate missing values (linear)

        Parameters
        ----------
        strategy : str, optional
            Strategy for dealing with missing values. The default is 'remove'.
        tol : float, optional
            Fraction of missing values for removing the whole station. The default is 0.1.
        
        Returns
        -------
        None.

        """
        print(f'Percentage of missing values:\n{100*self.ds.isna().sum()/self.ds.shape[0]}')
        if strategy=='stations':
            print(f'Removing stations with high percentage of missing values (tol={tol})')
            mask = self.ds.isna().sum()/self.ds.shape[0]>tol
            idx = [i[0] for i in np.argwhere(mask.values)]
            refst_remove = ['O3_'+self.RefStations[i] for i in idx]
            self.ds = self.ds.drop(columns=refst_remove)
            
        if strategy == 'remove':
            print('Removing missing values')
            self.ds.dropna(inplace=True)
            print(f'Entries with missing values remiaining:\n{self.ds.isna().sum()}')
            print(f'{self.ds.shape[0]} remaining measurements')
            
        elif strategy == 'interpolate':
            print('Interpolating missing data')
            self.ds = self.ds.interpolate(method='linear')
            print(f'Entries with missing values remiaining:\n{self.ds.isna().sum()}')
            
    def project_basis(self,Psi):
        """
        Project dataset onto lowrank-basis subspace defined by the columns of Psi
        The snapshots matrix (ds.T) is the one projected onto the subspace.
        To express it as dataset:
            ds_proj = (Psi@Psi.T@snapshots).T = (Psi@Psi.T@ds.T).T = ds@Psi@Psi.T
  
        Parameters
        ----------
        Psi : numpy array
            low-rank basis
  
        Returns
        -------
        self.ds_proj : pandas dataFrame
            projected dataset
  
        """
        snapshots_matrix = self.ds_train.T.values
        
        avg = np.mean(snapshots_matrix,axis=1)
        snapshots_matrix_centered = (snapshots_matrix - avg[:,None])
        self.ds_train_projected = pd.DataFrame((Psi@Psi.T@snapshots_matrix_centered).T,index = self.ds_train.index)
        
        snapshots_matrix = self.ds_val.T.values
        snapshots_matrix_centered = (snapshots_matrix - avg[:,None])
        self.ds_val_projected = pd.DataFrame((Psi@Psi.T@snapshots_matrix_centered).T,index = self.ds_val.index)
        
        snapshots_matrix = self.ds_test.T.values
        snapshots_matrix_centered = (snapshots_matrix - avg[:,None])
        self.ds_test_projected = pd.DataFrame((Psi@Psi.T@snapshots_matrix_centered).T,index = self.ds_test.index)
        
      
    def perturbate_signal(self,ds_signal,variance,seed):
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=0.0,scale=variance,size=ds_signal.shape)
        return  ds_signal + noise
            
if __name__ == '__main__':
    print('Testing')
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/'
    results_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Results/'
    
    pollutant = 'O3'
    start_date = '2011-01-01'
    end_date = '2022-12-31'
    
    dataset = DataSet(pollutant, start_date, end_date, files_path)
    dataset.load_dataSet()
    dataset.cleanMissingvalues(strategy='stations',tol=0.1)
    dataset.cleanMissingvalues(strategy='remove')
    

