#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:20:54 2023

@author: jparedes
"""
import pandas as pd
import numpy as np
# =============================================================================
# Low-rank basis of network
# =============================================================================
class LowRankBasis():
    def __init__(self,df,r):
        """
        measurements data frame

        Parameters
        ----------
        df : pandas data frame
            Measurements matrix. Shape (num_samples x num_stations)
        r: int
            Number of modes to use

        Returns
        -------
        None.

        """
        self.df = df
        self.r = r
        
    def snapshots_matrix(self):
        """
        Computes snapshots matrix.
        Each column of vector of snapshots matrix is the whole space at a give time.

        Returns
        -------
        None.
        """
        print('Rearranging data set to form snapshots matrix')
        self.snapshots_matrix  = self.df.T.values
        print(f'Snapshots matrix has dimensions {self.snapshots_matrix.shape}')
        
        
    
    def low_rank_decomposition(self,normalize=True):
        """
        Low-rank decomposition of snapshots matrix

        Parameters
        ----------
        normalize : bool, optional
            Whether or not to normalize data set. The default is True.

        """
        
        avg = np.mean(self.snapshots_matrix,axis=1)
        std = np.std(self.snapshots_matrix,axis=1)
        if normalize:
            print()
            X_ = (self.snapshots_matrix - avg[:,None])/(std[:,None])
        else:
            X_ = self.snapshots_matrix.copy()
            
        print(f'Snapshots matrix created: {X_.shape[1]} measurements of vector space {X_.shape[0]}')
        print(f'Matrix size in memory {X_.__sizeof__()} bytes')
        self.U, self.S, self.V = np.linalg.svd(X_,full_matrices=False)# full_matrices False for memory
        self.Psi = self.U[:,:self.r]
        
if __name__ == '__main__':
    print('Testing')
    n,m,r = 10,100,3
    rng = np.random.default_rng(seed=40)
    df = pd.DataFrame(rng.random((m,n)))
    
    lowrank_basis = LowRankBasis(df, r)
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    
    
        