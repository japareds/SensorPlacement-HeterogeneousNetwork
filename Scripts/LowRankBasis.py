#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 12:20:54 2023

@author: jparedes
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
        
        avg = np.mean(self.snapshots_matrix,axis=1)
        std = np.std(self.snapshots_matrix,axis=1)
        self.snapshots_matrix_centered = (self.snapshots_matrix - avg[:,None])#/(std[:,None])
        
    
    def low_rank_decomposition(self,normalize=True):
        """
        Low-rank decomposition of snapshots matrix

        Parameters
        ----------
        normalize : bool, optional
            Whether or not to normalize data set. The default is True.

        """
        
        if normalize:
            X_ = self.snapshots_matrix_centered.copy()
        else:
            X_ = self.snapshots_matrix.copy()
            
        print(f'Snapshots matrix created: {X_.shape[1]} measurements of vector space {X_.shape[0]}')
        print(f'Matrix size in memory {X_.__sizeof__()} bytes')
        self.U, self.S, self.V = np.linalg.svd(X_,full_matrices=False)# full_matrices False because of memory
        self.Psi = self.U[:,:self.r]
        
    def reconstruction_error(self,r,norm,snapshots_matrix_test):
        """
        Compute SVD reconstruction error on both training set and validation set.
        The class already has the snapshots matrix of the training set (used for computing SVD)

        Parameters
        ----------
        r : int
            low-rank order
        norm : str
            Norm computed. Options are ['RMSE','MAE','Frobenius']
        snapshots_matrix_test : pandas dataframe or numpy 2D array
            Test/validation set snapshots matrix

        Returns
        -------
        list
            Computed error in both training and validation set.[error_train,error_test]

        """
        
        # low-rank reconstruction
        #X_reconstructed = self.U[:,:r]@np.diag(self.S[:r])@self.V[:r,:]
        X_reconstructed_train = self.U[:,:r]@self.U[:,:r].T@self.snapshots_matrix_centered
        
        avg = np.mean(self.snapshots_matrix,axis=1)
        std = np.std(self.snapshots_matrix,axis=1)
        snapshots_matrix_test_centered =  (snapshots_matrix_test - avg[:,None])/(std[:,None])
        X_reconstructed_test =  self.U[:,:r]@self.U[:,:r].T@snapshots_matrix_test_centered
        
        if norm=='RMSE':
            error_train = np.sqrt(mean_squared_error(self.snapshots_matrix_centered, X_reconstructed_train))
            error_test = np.sqrt(mean_squared_error(snapshots_matrix_test_centered,X_reconstructed_test))
        elif norm == 'MAE':
            error_train = mean_absolute_error(self.snapshots_matrix_centered, X_reconstructed_train)
            error_test = mean_absolute_error(snapshots_matrix_test_centered,X_reconstructed_test)
        elif norm == 'Frobenius':
            error_train = np.linalg.norm(self.snapshots_matrix_centered - X_reconstructed_train,ord='fro')
            error_test = np.linalg.norm(snapshots_matrix_test_centered - X_reconstructed_test,ord='fro')
            
        
        return [error_train,error_test]
    
if __name__ == '__main__':
    print('Testing')
    n,m,r = 10,100,3
    rng = np.random.default_rng(seed=40)
    df = pd.DataFrame(rng.random((m,n)))
    df_test = pd.DataFrame(rng.random((m,n)))
    
    lowrank_basis = LowRankBasis(df, r)
    lowrank_basis.snapshots_matrix()
    lowrank_basis.low_rank_decomposition(normalize=True)
    
    error_train = {el:0 for el in range(n)}
    error_test = {el:0 for el in range(n)}
    for r_ in range(n):
        error_train,error_test = lowrank_basis.reconstruction_error(r_, 'RMSE', df_test.T)
    
    
        