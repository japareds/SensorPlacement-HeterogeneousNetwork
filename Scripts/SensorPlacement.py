#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:21:04 2023

@author: jparedes
"""
import numpy as np
import cvxpy as cp
from scipy import linalg
import warnings
# =============================================================================
# Sensor placement algorithms
# =============================================================================
class SensorsErrors(Exception):
    pass

class SensorPlacement:
    def __init__(self,algorithm,n,r,p_zero,p_eps,p_empty,var_eps,var_zero):
        """
        Network parameters

        Parameters
        ----------
        algorithm : str
            Sensor placement strategy: D_optimal, E_optimal, WCS
        n : int
            maximum number of possible locations
        r : int
            number of basis vectors
        p_zero : int
            number of reference stations
        p_eps : int
            number of LCSs
        p_empty : int
            number of empty locations (unmonitored)
         var_eps : float
             variance LCSs
         var_zero : float
             variance Reference stations

        Returns
        -------
        None.

        """
        self.algorithm = algorithm
        self.n = n
        self.r = r
        self.p_zero = p_zero
        self.p_eps = p_eps
        self.p_empty = p_empty
        self.var_eps = var_eps
        self.var_zero = var_zero
    
    def WCS_placement(self,Psi):
        """
        Worst case scenario sensor placement algorithm
        Solve PSD to minimize maximum diagonal entry of covariance matrix

        Parameters
        ----------
        Psi : np array
            reduced basis
       

        Returns
        -------
        None.
        self.problem added to class

        """
        # compute covariance matrix
        h_eps = cp.Variable(shape=self.n,value=np.zeros(self.n))
        h_zero = cp.Variable(shape = self.n,value=np.zeros(self.n))
        t = cp.Variable(shape=(1,1))
        
        #S = np.zeros((self.r,self.r))
        # precision matrix as sum of both: LCSs and ref.st
        # for i in range(self.n):
        #     psi = Psi[i,:][None,:]
        #     S+=h_eps[i]*(var_eps**-1)*psi.T@psi + h_zero[i]*(var_zero**-1)*psi.T@psi 
        if self.p_eps != 0:
            S = cp.sum([h_eps[i]*(self.var_eps**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] + h_zero[i]*(self.var_zero**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)])
        else:
            S = cp.sum([h_zero[i]*(self.var_zero**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)])
        
        constraints = [t>=0,
                       cp.sum(h_zero)==self.p_zero,
                       h_zero >= 0,
                       h_zero <=1]
        
        if self.p_eps !=0:
            constraints += [cp.sum(h_eps)==self.p_eps,
                            h_eps >= 0,
                            h_eps <=1,
                            h_eps + h_zero <=1]
 
        
        Ir = np.identity(self.r)
        # PSD constraints: r block matrices
        for j in np.arange(self.r):
            constraints += [cp.bmat([[t,Ir[:,j][:,None].T],[Ir[:,j][:,None],S]]) >> 0]
        problem = cp.Problem(cp.Minimize(t),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        
        self.h_eps = h_eps
        self.h_zero = h_zero
        self.problem = problem
    
    def unmonitored_placement(self,Psi,alpha):
        """
        Sensor placement proposed heuristics for unmonitored locations (p_empty !=0)
        Minimize the rank for reference stations locations while minimizing volume of ellipsoid for LCSs

        Parameters
        ----------
        Psi : np array
            Reduced basis of shape (number of locations, number of vectors)
        alpha : flaot
            regularization parameter for rank minimization constraint

        Returns
        -------
        None.

        """
        h_eps = cp.Variable(shape=self.n,value=np.zeros(self.n))
        h_zero = cp.Variable(shape = self.n,value=np.zeros(self.n))
        objective_eps = cp.log_det(cp.sum([h_eps[i]*(self.var_eps**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
        objective_zero = alpha*cp.trace(cp.sum([h_zero[i]*(self.var_zero**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
        objective = objective_eps + objective_zero
        constraints = [cp.sum(h_zero) == self.p_zero,
                       cp.sum(h_eps) == self.p_eps,
                       h_zero >=0,
                       h_zero <= 1,
                       h_eps >=0,
                       h_eps <=1,
                       h_eps + h_zero <=1]
        problem = cp.Problem(cp.Maximize(objective),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h_eps = h_eps
        self.h_zero = h_zero
        self.problem = problem
        
    def DOptimal_placement(self,Psi):
        """
        D-optimal sensor placement for two classes of sensors with variances var_zero and var_eps
        Maximize logdet of precision matrix
        
        Parameters
        ----------
        Psi : np array
            Reduced basis of shape (number of locations, number of vectors)
            
        Returns
        -------
        None.

        """
        h_zero = cp.Variable(shape = self.n,value=np.zeros(self.n))
        h_eps = cp.Variable(shape=self.n,value=np.zeros(self.n))
        
        if self.p_eps != 0 and self.p_zero != 0: # there are LCSs and ref st.
            objective = cp.log_det(cp.sum([h_eps[i]*(self.var_eps**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] + h_zero[i]*(self.var_zero**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [cp.sum(h_zero) == self.p_zero,
                           cp.sum(h_eps) == self.p_eps,
                           h_zero >=0,
                           h_zero <= 1,
                           h_eps >=0,
                           h_eps <=1,
                           h_eps + h_zero <=1]
            
            
        elif self.p_eps == 0 and self.p_zero != 0:# there are no LCSs
            objective = cp.log_det(cp.sum([h_zero[i]*(self.var_zero**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [cp.sum(h_zero) == self.p_zero,
                           h_zero >=0,
                           h_zero <= 1,
                           ]
            
        
        elif self.p_eps != 0 and self.p_zero == 0:# there are no ref.st.
            objective = cp.log_det(cp.sum([h_eps[i]*(self.var_eps**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [cp.sum(h_eps) == self.p_eps,
                           h_zero >=0,
                           h_zero <= 1,
                           ]
            
            
        problem = cp.Problem(cp.Maximize(objective),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h_eps = h_eps
        self.h_zero = h_zero
        self.problem = problem
        
    
    def check_consistency(self):
        """
        Check number of sensors consistency 

        Raises
        ------
        SensorsErrors
            Error if the proposed configuration is not correct

        Returns
        -------
        None.

        """
        
        # check number of sensors consistency
        if self.p_zero >self.r and self.p_eps!=0:
            raise SensorsErrors(f'The number of reference stations is larger than the dimension of the low-rank basis {self.r}')
        if any(np.array([self.p_eps,self.p_zero,self.p_empty])<0):
            raise SensorsErrors('Negative number of sensors')
        if all(np.array([self.p_eps,self.p_zero,self.p_empty])>=0) and np.sum([self.p_eps,self.p_zero,self.p_empty]) != self.n:
            raise SensorsErrors('Number of sensors and empty locations mismatch total number of possible locations')
        
        
    def initialize_problem(self,Psi,alpha=0):
        
        if self.algorithm == 'WCS':
            self.WCS_placement(Psi)
        
        elif self.algorithm == 'unmonitored':
            if self.p_empty == 0:
                warnings.warn(f'Sensor placement algorithm is {self.algorithm} but there are {self.p_empty} unmonitored locations.')
                input('Press Enter to continue...')
                self.unmonitored_placement(Psi,alpha)
            else:
                self.unmonitored_placement(Psi,alpha)
        
        elif self.algorithm == 'D_optimal':
            self.DOptimal_placement(Psi)
            
        else:
            print(f'Sensor placement algorithm {self.algorithm} not implemented yet')
        
        self.check_consistency()
        
        
    
    def solve(self):
        """
        Solve sensor placement problem and print objective function optimal value
        and sensors weights

        Returns
        -------
        None.

        """
        print(f'Solving sensor placement using {self.algorithm} strategy')
        if self.algorithm == 'WCS':
            solver = cp.MOSEK
            try:
                self.problem.solve(verbose=True,solver=solver)
                if self.problem.status not in ["infeasible", "unbounded"]:
                    print(f'Optimal value: {self.problem.value}')
            except:
                print('Problem not solved.\nIt is possible solver failed or problem status is unknown.')
        else:
            try:
                self.problem.solve(verbose=True)
                if self.problem.status not in ["infeasible", "unbounded"]:
                    print(f'Optimal value: {self.problem.value}')
            except:
                print('Problem not solved.\nIt is possible solver failed or problem status is unknown.')
        # print(f'Weights disitrubtion\nLCSs:')
        # print(*np.round(self.h_eps.value,3),sep='\n')
        # print(f'Reference stations:')
        # print(*np.round(self.h_zero.value,3),sep='\n')
        
    def convertSolution(self):
        """
        Get maximum entries of h and use those locations to place sensors

        Returns
        -------
        None.

        """
        if self.p_zero == 0:
            order_zero = []
            order_eps = np.sort(np.argsort(self.h_eps.value)[-self.p_eps:])
        elif self.p_eps == 0:
            order_zero = np.sort(np.argsort(self.h_zero.value)[-self.p_zero:])
            order_eps = []
        else:
            order_zero = np.sort(np.argsort(self.h_zero.value)[-self.p_zero:])
            order_eps = np.sort([i for i in np.flip(np.argsort(self.h_eps.value)) if i not in order_zero][:self.p_eps])
        
        order_empty = np.sort([i for i in np.arange(self.n) if i not in order_eps and i not in order_zero])
        
        if (set(order_eps).issubset(set(order_zero)) or set(order_zero).issubset(set(order_eps))) and (self.p_eps!=0 and self.p_zero!=0) :
            print('Some locations of LCSs are already occupied by a reference station')
            order_eps = [i for i in np.arange(self.n) if i not in order_zero]
            
        self.locations = [order_eps,order_zero,order_empty]
        
    def C_matrix(self):
        """
        Convert indexes of LCSs, RefSt and Emtpy locations into
        C matrix

        Returns
        -------
        self.C: list
            C matrix for LCSs, RefSt, Empty
            

        """
        In = np.identity(self.n)
        C_eps = In[self.locations[0],:]
        C_zero = In[self.locations[1],:]
        if self.p_empty != 0:
            C_empty = In[self.locations[2],:]
        else:
            C_empty = []
        self.C = [C_eps,C_zero,C_empty]

    def covariance_matrix(self,Psi,metric,alpha=0.1):
        """
        Compute covariance matrix from C matrices

        Parameters
        ----------
        Psi : np array
            low-rank basis used
        metric : str
            metric to compute covariance optimizality: D-optimal, E-optimal, WCS

        Returns
        -------
        None.

        """
        C_eps = self.C[0]
        C_zero = self.C[1]
        
        Theta_eps = C_eps@Psi
        Theta_zero = C_zero@Psi
        if self.p_eps !=0:
        
            try:
                self.Cov = np.linalg.inv( (self.var_eps**(-1)*Theta_eps.T@Theta_eps) + (self.var_zero**(-1)*Theta_zero.T@Theta_zero) ) 
            except:
                print('Computing pseudo-inverse')
                self.Cov = np.linalg.pinv( (self.var_eps**(-1)*Theta_eps.T@Theta_eps) + (self.var_zero**(-1)*Theta_zero.T@Theta_zero) )
        
        else:
            try:
                self.Cov = np.linalg.inv(self.var_zero**(-1)*Theta_zero.T@Theta_zero)
            except:
                print('Computing pseudo-inverse')
                self.Cov = np.linalg.pinv(self.var_zero**(-1)*Theta_zero.T@Theta_zero)
        
        if metric == 'D_optimal':
            self.metric = np.log(np.linalg.det(self.Cov))
        elif metric == 'E_optimal':
            self.metric = np.max(np.real(np.linalg.eig(self.Cov)[0]))
        elif metric == 'WCS':
            self.metric = np.diag(self.Cov).max()
        elif metric == 'Proposed_algorithm':
            self.metric = np.log(np.linalg.det(self.var_eps**(-1)*Theta_eps.T@Theta_eps)) + alpha*np.trace(self.var_zero**(-1)*Theta_zero.T@Theta_zero)
            
        
        if type(self.problem.value) == type(None):# error when solving
            self.metric = -1
        
        
        
if __name__ == '__main__':
    p_eps,p_zero,p_empty,n,r = 2,3,1,6,3
    var_eps,var_zero = 1,1e-2
    rng = np.random.default_rng(seed=40)
    U = linalg.orth(rng.random((n,n)))
    Psi = U[:,:r]
    algorithm = 'WCS'
    sensor_placement = SensorPlacement(algorithm, n, r, p_zero, p_eps, p_empty, var_eps, var_zero)
    sensor_placement.initialize_problem(Psi)
    sensor_placement.solve()
    sensor_placement.convertSolution()
    sensor_placement.C_matrix()
    sensor_placement.covariance_matrix(Psi)