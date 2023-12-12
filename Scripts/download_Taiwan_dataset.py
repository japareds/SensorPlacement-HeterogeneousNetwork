#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:23:08 2023

@author: jparedes
"""

import pandas as pd
import pickle
import numpy as np
from urllib import request
import shutil
import os

#%% year of measurements and url
# dataset obtained from Ministry of Environment_National Air Quality Testing Station:
# https://history.colife.org.tw/#/?cd=%2F%E7%A9%BA%E6%B0%A3%E5%93%81%E8%B3%AA%2F%E7%92%B0%E5%A2%83%E9%83%A8_%E5%9C%8B%E5%AE%B6%E7%A9%BA%E5%93%81%E6%B8%AC%E7%AB%99

dict_urls = {2013:'https://history.colife.org.tw/?r=/download&path=L%2Bepuuawo%2BWTgeizqi%2FnkrDlooPpg6hf5ZyL5a6256m65ZOB5ris56uZL01PRU5WX0VSREJfMTk5OC0yMDEzLnppcA%3D%3D',
             2014:'https://history.colife.org.tw/?r=/download&path=L%2Bepuuawo%2BWTgeizqi%2FnkrDlooPpg6hf5ZyL5a6256m65ZOB5ris56uZL01PRU5WX0VSREJfMjAxNC56aXA%3D',
             2015:'https://history.colife.org.tw/?r=/download&path=L%2Bepuuawo%2BWTgeizqi%2FnkrDlooPpg6hf5ZyL5a6256m65ZOB5ris56uZL01PRU5WX0VSREJfMjAxNS56aXA%3D',
             2016:'https://history.colife.org.tw/?r=/download&path=L%2Bepuuawo%2BWTgeizqi%2FnkrDlooPpg6hf5ZyL5a6256m65ZOB5ris56uZL01PRU5WX0VSREJfMjAxNi56aXA%3D',
             2017:'https://history.colife.org.tw/?r=/download&path=L%2Bepuuawo%2BWTgeizqi%2FnkrDlooPpg6hf5ZyL5a6256m65ZOB5ris56uZL01PRU5WX09EXzIwMTcuemlw',
             2018:'https://history.colife.org.tw/?r=/download&path=L%2Bepuuawo%2BWTgeizqi%2FnkrDlooPpg6hf5ZyL5a6256m65ZOB5ris56uZL01PRU5WX09EXzIwMTguemlw',
             2019:'https://history.colife.org.tw/?r=/download&path=L%2Bepuuawo%2BWTgeizqi%2FnkrDlooPpg6hf5ZyL5a6256m65ZOB5ris56uZL01PRU5WX09EXzIwMTkuemlw',
             2020:'https://history.colife.org.tw/?r=/download&path=L%2Bepuuawo%2BWTgeizqi%2FnkrDlooPpg6hf5ZyL5a6256m65ZOB5ris56uZL01PRU5WX09EXzIwMjAuemlw',
             2021:'https://history.colife.org.tw/?r=/download&path=L%2Bepuuawo%2BWTgeizqi%2FnkrDlooPpg6hf5ZyL5a6256m65ZOB5ris56uZL01PRU5WX09EXzIwMjEuemlw',
             2022:'https://history.colife.org.tw/?r=/download&path=L%2Bepuuawo%2BWTgeizqi%2FnkrDlooPpg6hf5ZyL5a6256m65ZOB5ris56uZL01PRU5WX09EXzIwMjIuemlw'
             }

#%%
class File():
    def __init__(self,url,fname):
        self.url = url
        self.fname = fname
    def download_from_url(self):
        request.urlretrieve(self.url,self.fname)
        with request.urlopen(self.url) as response, open(fname, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
        print(f'Downloaded file {self.fname}')
        
class DataSet():
    def __init__(self,year,files_path,pollutant):
        self.year = year
        self.path = files_path+f'dataset{year}/'
        self.df = pd.DataFrame()
        self.pollutant = pollutant
    
    def load_dataset(self):
        if self.year in [2022,2021]:# many loose csv files
            months = np.arange(1,13,1)
            for m in months:
                print(f'Loading month {m}')
                fname = self.path+f'EPA_OD_{self.year}{m:02}.csv'
                try:
                    df_ = pd.read_csv(fname)
                    self.df = pd.concat([self.df,df_],axis=0)
                except:
                    print(f'No file for month {m} and year {self.year}')
        # single files
        if self.year in [2020,2019,2018,2017]:
            fname = self.path+f'epa_{self.year}.csv'
            self.df = pd.read_csv(fname)
        
        
    def explore_multiple_locations(self):
        """
        Some locations_id are repeated for different locations (lat,lon).
        Explore those sensors and locations.

        Returns
        -------
        None.

        """
        print(f'dataset {self.year}\nnumber of sites: {self.df.SiteId.unique().shape[0]}\nNumber of Latitudes,Longitudes: ({self.df.Latitude.unique().shape[0]},{self.df.Longitude.unique().shape[0]})')
        sites = np.sort(self.df.SiteId.unique())
        for s in sites:
            df = self.df.loc[self.df.SiteId==s]
            latitudes = df.Latitude.unique()
            longitudes = df.Longitude.unique()
            if latitudes.shape[0]>1:
                print(f'\nLocation Id {s} has {df.shape[0]} measurements in total.')
                print(f'\t{latitudes.shape[0]} different latitudes: {latitudes}')
                for l in latitudes:
                    df_lat = df.loc[df.Latitude==l]
                    print(f'Latitude {l} has {df_lat.shape[0]} measurements')
            if longitudes.shape[0]>1:
                print(f'Location Id {s} has {df.shape[0]} measurements in total')
                print(f'\t{longitudes.shape[0]} different longitudes: {longitudes}')
                for l in longitudes:
                    df_lon = df.loc[df.Longitude==l]
                    print(f'Longitude {l} has {df_lon.shape[0]} measurements')
    
    def extract_columns(self):
        """
        Keep certain columns from dataset and reformat

        Returns
        -------
        None.

        """
        self.df.drop_duplicates(subset=['PublishTime','SiteId'],inplace=True)
        self.df = self.df.loc[:,[self.pollutant,'PublishTime','SiteId','Latitude','Longitude']]
        self.df.SiteId = self.df.SiteId.astype('Int64')
        self.df.set_index('PublishTime',drop=True,inplace=True)
        self.df.index.name = 'date'
        self.df.index = pd.to_datetime(self.df.index)
        if self.pollutant == 'O3':
            self.df.loc[self.df.O3=='-','O3'] = np.nan
        elif self.pollutant == 'NO2':
            self.df.loc[self.df.NO2=='-','NO2'] = np.nan
            
    def save_dataset(self):
        self.df.to_csv(f'{self.path}{self.pollutant}_{self.year}.csv')
        print(f'{self.pollutant} pollutant for year {self.year} saved at {self.path}')
    
                
                
    def load_all_datasets(self,files_path,years):
        self.dict_df = {el:[] for el in years}
        for y in years:
            fname = f'{files_path}dataset{y}/{self.pollutant}_{y}.csv'
            df = pd.read_csv(fname)
            self.dict_df[y] = df
    
    def get_shared_coordinates(self,years):
        """
        Get shared coordinates (lat,lon) monitored accross all years

        Parameters
        ----------
        years : list
            years with data

        Returns
        -------
        None.

        """
        self.dict_coords = {el:[] for el in years}
        
        for y in years:
            df = self.dict_df[y]
            dict_ = dict([*df.groupby(['Latitude','Longitude'])])
            self.dict_coords[y] = [i for i in dict_.keys()]
            
        year_orig = years[0]
        coords_shared = np.array(self.dict_coords[year_orig])
        for y in years[1:]:
            print(f'Comparing year {y}')
            coords = np.array(self.dict_coords[y])
            mask = np.isin(coords_shared,coords)
            idx_mismatch = np.argwhere(np.sum(mask,axis=1)==1)
            if len(idx_mismatch) >0:
                mask[idx_mismatch[0]] = [False,False]
            coords_shared = np.reshape(coords_shared[mask],(int(coords_shared[mask].shape[0]/2),2))
            #coords_shared = np.array([[i,j] for i,j in zip(coords_shared[mask[:,0],0],coords_shared[mask[:,1],1])])
        self.coords_shared = coords_shared    
            
    def reduce_dataframe(self,years):
        """
        Keep measurements at locations monitored during the whole period in years

        Parameters
        ----------
        years : list
            years of measurements

        Returns
        -------
        None.

        """
        f = lambda x:np.argwhere(self.coords_shared[:,0]==x)[0][0]
        
        for y in years:
            print(f'Reducing dataset year {y}')
            df = self.dict_df[y]
            df = df.loc[df.Latitude.isin(self.coords_shared[:,0])].loc[df.Longitude.isin(self.coords_shared[:,1])]
            df['Station_number'] = [f(i) for i in df.Latitude]
            # df.set_index(df.date,inplace=True)
            # df.sort_index(inplace=True)
            self.dict_df[y] = df
            
    def sort_stations(self,years):
        """
        Sort dataset to show time series for each station.
        use date as index and each column represents pollutant measurement for each station

        Parameters
        ----------
        years : list
            years of campaign

        Returns
        -------
        None.

        """
        self.df = {el:[] for el in years}
        for y in years:
            print(f'Sorting year: {y}')
            df = self.dict_df[y]
            df = df.groupby(by=['Station_number','date',f'{self.pollutant}'],as_index=False).mean()
            stations_id = np.sort(df.Station_number.unique())
            stations_id = stations_id[~np.isnan(stations_id)]
            
            i=0
            df_station = pd.DataFrame(df.loc[df.Station_number==i,[f'{self.pollutant}','date']])
            df_station.drop_duplicates(subset='date',inplace=True)
            df_station.set_index('date',inplace=True,drop=True)
            df_station.sort_index(inplace=True)
            df_station.index = pd.to_datetime(df_station.index)
            df_station.columns = [f'{self.pollutant}_station_{i}']
            df_station = df_station.apply(pd.to_numeric)
            
            for i in stations_id[1:]:
                df_ = pd.DataFrame(df.loc[df.Station_number==i,[f'{self.pollutant}','date']])
                df_.drop_duplicates(subset='date',inplace=True)
                df_.set_index('date',inplace=True,drop=True)
                df_.sort_index(inplace=True)
                df_.index = pd.to_datetime(df_.index)
                df_.columns = [f'{self.pollutant}_station_{i}']
                df_ = df_.apply(pd.to_numeric)
            
                df_station = pd.concat([df_station,df_],axis=1)
            
            self.df[y] = df_station
            
    def getStations_coordinates(self,years,path):
        
        coordinates = self.coords_shared
        self.coordinates_sorted = {el:[] for el in years}
        for y in years:
            df = self.dict_df[y]
            stations_numbers = df.loc[:,'Station_number'].unique()
            for i in stations_numbers:
                self.coordinates_sorted[y].append(np.array([df.loc[df.Station_number == i].Latitude.unique(),df.loc[df.Station_number == i].Longitude.unique()]))
            self.coordinates_sorted[y] = np.reshape(np.array(dataset.coordinates_sorted[y]),(len(dataset.coordinates_sorted[y]),2))
        fname = f'{path}Coordinates_stations_{self.pollutant}_Taiwan_{years[0]}_{years[-1]}.csv'
        with open(fname,'wb') as f:
            pickle.dump(self.coordinates_sorted,f)
            
    def join_years(self,years):
        """
        Concatenate all yearly measurements to generate single dataset

        Parameters
        ----------
        years : list
            list of years with measurements

        Returns
        -------
        None.

        """
        df_full = pd.DataFrame()
        
        if years[0]>years[1]:
            years = np.flip(years)
        
        for y in years:
            df_full= pd.concat([df_full,self.df[y]],axis=0)
        self.ds = df_full
            
    

    
#%%
if __name__=='__main__':    
    abs_path = os.path.dirname(os.path.realpath(__file__))
    files_path = os.path.abspath(os.path.join(abs_path,os.pardir)) + '/Files/Taiwan/'
    
    download_files = False
    if download_files:
        print(f'Files will be downloaded into {files_path}')
        input('Press Enter to continue ...')
        for year in dict_urls:
            fname = files_path+f'dataset{year}.zip'
            kfile = File(dict_urls[year],fname)
            kfile.download_from_url()
   
    preprocess_dataset = False
    if preprocess_dataset:
        print('Pre-processing dataset')
        year = 2018# [2022,2021,2020,2019,2018]
        pollutant = 'NO2' #['O3', 'NO2']
        dataset = DataSet(year, files_path,pollutant)
        dataset.load_dataset()
        dataset.explore_multiple_locations()
        dataset.extract_columns()
        #dataset.save_dataset()
    
    join_datasets = True
    if join_datasets:
        pollutant = 'O3' #['O3', 'NO2']
        years = [2022,2021,2020,2019,2018]
        dataset = DataSet(years,files_path,pollutant)
        dataset.load_all_datasets(files_path, years)
        dataset.get_shared_coordinates(years)
        dataset.reduce_dataframe(years)
        dataset.sort_stations(years)
        dataset.getStations_coordinates(years,files_path)
        dataset.join_years(years)
        
            
