#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 17:21:04 2023

@author: jparedes
"""
import numpy as np
import cvxpy as cp
from scipy import linalg
import pickle
import time
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
# =============================================================================
#         Sensor placement algorithms
# =============================================================================
    
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
    
    def rankMax_placement(self,Psi,alpha,substract=False):
        """
        Sensor placement proposed heuristics for unmonitored locations (p_empty !=0)
        Maximize the rank for reference stations locations while minimizing volume of ellipsoid for LCSs

        Parameters
        ----------
        Psi : np array
            Reduced basis of shape (number of locations, number of vectors)
        alpha : float
            regularization parameter for rank minimization constraint

        Returns
        -------
        None.

        """
        
        h_eps = cp.Variable(shape=Psi.shape[0],value=np.zeros(Psi.shape[0]))
        h_zero = cp.Variable(shape = Psi.shape[0],value=np.zeros(Psi.shape[0]))
        objective_eps = -1*cp.log_det(cp.sum([h_eps[i]*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(Psi.shape[0])]))
        if substract:
            print('Objective function for trace will be negative: logdet(LCS) - alpha*Tr(RefSt)')
            objective_zero = -alpha*cp.trace(cp.sum([h_zero[i]*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(Psi.shape[0])]))
        else:
            objective_zero = alpha*cp.trace(cp.sum([h_zero[i]*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(Psi.shape[0])]))
            
        objective = objective_eps + objective_zero
        constraints = [cp.sum(h_zero) == self.p_zero,
                       cp.sum(h_eps) == self.p_eps,
                       h_zero >=0,
                       h_zero <= 1,
                       h_eps >=0,
                       h_eps <=1,
                       h_eps + h_zero <=1]
        problem = cp.Problem(cp.Minimize(objective),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h_eps = h_eps
        self.h_zero = h_zero
        self.problem = problem
        

    def simple_Dopt_placement(self,Psi):
        """
        Simple D-optimal sensor placement for single class of sensors

        Parameters
        ----------
        Psi : np array
            Reduced basis of shape (number of locations, number of vectors)
        alpha : float
            regularization parameter for rank minimization constraint

        Returns
        -------
        None.

        """
        h = cp.Variable(shape=self.n,value=np.zeros(self.n))
        objective = -1*cp.log_det(cp.sum([h[i]*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
        constraints = [cp.sum(h) == self.p_eps + self.p_zero,
                       h >=0,
                       h <= 1,
                       ]
        problem = cp.Problem(cp.Minimize(objective),constraints)
        
        
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h = h
        self.problem_1 = problem
        
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
            objective = -1*cp.log_det(cp.sum([h_eps[i]*(self.var_eps**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] + h_zero[i]*(self.var_zero**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [cp.sum(h_zero) == self.p_zero,
                           cp.sum(h_eps) == self.p_eps,
                           h_zero >=0,
                           h_zero <= 1,
                           h_eps >=0,
                           h_eps <=1,
                           h_eps + h_zero <=1]
            
            
        elif self.p_eps == 0 and self.p_zero != 0:# there are no LCSs
            objective = -1*cp.log_det(cp.sum([h_zero[i]*(self.var_zero**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [cp.sum(h_zero) == self.p_zero,
                           h_zero >=0,
                           h_zero <= 1,
                           ]
            
        
        elif self.p_eps != 0 and self.p_zero == 0:# there are no ref.st.
            objective = -1*cp.log_det(cp.sum([h_eps[i]*(self.var_eps**-1)*Psi[i,:][None,:].T@Psi[i,:][None,:] for i in range(self.n)]))
            constraints = [cp.sum(h_eps) == self.p_eps,
                           h_eps >=0,
                           h_eps <= 1,
                           ]
            
            
        problem = cp.Problem(cp.Minimize(objective),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.h_eps = h_eps
        self.h_zero = h_zero
        self.problem = problem
        
    def Liu_placement_uncorr(self,Psi):
        """
        Liu-based sensor placement for weakly-correlated noise sensors

        Parameters
        ----------
        Psi : numpy array
            low-rank basis

        Returns
        -------
        None.

        """
        R = np.identity(self.n)
        diag = np.concatenate((np.repeat(self.var_zero**-1,self.p_zero),
                               np.repeat(self.var_eps**-1,self.p_eps),
                               np.repeat(0,self.p_empty)))
        np.fill_diagonal(R, diag)
        h = cp.Variable(shape=self.n,value = np.zeros(self.n))
        H = cp.Variable(shape=(self.n,self.n),value=np.zeros((self.n,self.n)))
        
        F_mat = Psi.T@(cp.multiply(H, R))@Psi
        
        Z = cp.Variable(shape=(self.r,self.r))
        Ir = np.identity(self.r)
        M = cp.bmat([F_mat,Ir],[Ir,Z])
        
        objective = cp.trace(Z)
        
        
        constraints = [M >= 0,
                       cp.trace(H)<= self.p_zero + self.p_eps,
                       cp.diag(H) == h,
                       cp.bmat([ [H,h[:,None]],[h[:,None].T,np.ones((1,1))] ] )>= 0]
        
        
        problem = cp.Problem(cp.Minimize(objective),constraints)
        if not problem.is_dcp():
            print('Problem not dcp\nCheck constraints and objective function')
        self.problem = problem
        self.h = h
      
       
# =============================================================================
#        Problem initialization
# =============================================================================
        
    
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
        # if self.p_zero >self.r and self.p_eps!=0:
        #     raise SensorsErrors(f'The number of reference stations is larger than the dimension of the low-rank basis {self.r}')
        if self.p_empty + self.r > self.n:
            raise SensorsErrors('Not enough sensors for basis sampling')
        if any(np.array([self.p_eps,self.p_zero,self.p_empty])<0):
            raise SensorsErrors('Negative number of sensors')
        if all(np.array([self.p_eps,self.p_zero,self.p_empty])>=0) and np.sum([self.p_eps,self.p_zero,self.p_empty]) != self.n:
            raise SensorsErrors('Number of sensors and empty locations mismatch total number of possible locations')
        
        
    def initialize_problem(self,Psi,alpha=0):
        
        if self.algorithm == 'WCS':
            self.WCS_placement(Psi)
        
        elif self.algorithm == 'rankMax':
            if self.p_empty == 0:
                self.rankMax_placement(Psi,alpha)
            elif self.p_eps == 0: # no LCS - use Doptimal
                self.var_zero = 1e0
                self.DOptimal_placement(Psi)
            else:
                self.rankMax_placement(Psi,alpha)
        
        elif self.algorithm == 'D_optimal':
            self.DOptimal_placement(Psi)
            
        elif self.algorithm == 'rankMax_FM':
            self.simple_Dopt_placement(Psi)
            self.Psi_phase1 = Psi.copy()
            self.alpha_phase2 = alpha
        
        elif self.algorithm == 'Dopt-Liu':
            self.Liu_placement_uncorr(Psi)
            
        else:
            print(f'Sensor placement algorithm {self.algorithm} not implemented yet')
        
        self.check_consistency()
        
# =============================================================================
#             Solve sensor problem
# =============================================================================
    def LoadLocations(self,dicts_path,alpha_reg,var_zero):
        """
        Load locations from previous training

        Parameters
        ----------
        dicts_path : str
            path to files
        alpha_reg : flaot
            regularization value
        var_zero : float
            refst covariance
            
        Returns
        -------
        self.dict_results : dict
            (LCSs, ref.st,empty) locations

        """
        if self.algorithm == 'rankMax' or self.algorithm == 'rankMax_FM' :
            fname = dicts_path+f'DiscreteLocations_{self.algorithm}_vs_p0_{self.n}N_r{self.r}_pEmpty{self.p_empty}_alpha{alpha_reg:.1e}.pkl'
            with open(fname,'rb') as f:
                self.dict_locations = pickle.load(f)
            
            fname = dicts_path+f'Weights_{self.algorithm}_vs_p0_{self.n}N_r{self.r}_pEmpty{self.p_empty}_alpha{alpha_reg:.1e}.pkl'
            with open(fname,'rb') as f:
                self.dict_weights = pickle.load(f)
        else:
            fname = dicts_path+f'DiscreteLocations_{self.algorithm}_vs_p0_{self.n}N_r{self.r}_pEmpty{self.p_empty}_varZero{var_zero:.1e}.pkl'
            with open(fname,'rb') as f:
                self.dict_locations = pickle.load(f)
            
            fname = dicts_path+f'Weights_{self.algorithm}_vs_p0_{self.n}N_r{self.r}_pEmpty{self.p_empty}_varZero{var_zero:.1e}.pkl'
            with open(fname,'rb') as f:
                self.dict_weights = pickle.load(f)
        
    def solve(self):
        """
        Solve sensor placement problem and print objective function optimal value
        and sensors weights

        Returns
        -------
        None.

        """
        time_init = time.time()
        print(f'Solving sensor placement using {self.algorithm} strategy')
        if self.algorithm == 'WCS':
            solver = cp.MOSEK
            try:
                self.problem.solve(verbose=True,solver=solver)
                if self.problem.status not in ["infeasible", "unbounded"]:
                    print(f'Optimal value: {self.problem.value}')
            except:
                print('Problem not solved.\nIt is possible solver failed or problem status is unknown.')
        
        elif self.algorithm == 'rankMax_FM':
            try:
                # phase 1) solve simple D-optimal to determine unmonitored locations
                print('Solving basic D-optimal problem')
                self.problem_1.solve(verbose=True)
                if self.problem_1.status not in ["infeasible", "unbounded"]:
                    print(f'Optimal value: {self.problem_1.value}')
                # phase2) solve fully monitored rankMax to distribute sensors
                self.loc_unmonitored = np.sort(np.argsort(self.h.value)[:self.p_empty])
                self.loc_monitored = np.sort(np.argsort(self.h.value)[-(self.p_zero + self.p_eps):])
                Psi_restricted = self.Psi_phase1[self.loc_monitored,:]
                self.rankMax_placement(Psi_restricted, self.alpha_phase2)
                self.problem.solve(verbose=True)
                print('Solving rankMax for sensor distribution')
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
                
        time_end = time.time()
        self.exec_time = time_end - time_init
        # print(f'Weights disitrubtion\nLCSs:')
        # print(*np.round(self.h_eps.value,3),sep='\n')
        # print(f'Reference stations:')
        # print(*np.round(self.h_zero.value,3),sep='\n')
        
    def discretize_solution(self):
        """
        Convert continuous weights [0,1] to discrete values {0,1}
        Get maximum entries of h and use those locations to place sensors

        Returns
        -------
        None.

        """
        
        if self.algorithm == 'rankMax_FM':
            order_zero = self.loc_monitored[np.sort(np.argsort(self.h_zero.value)[-self.p_zero:])]
            order_eps = np.sort(np.setdiff1d(self.loc_monitored,order_zero))
            order_empty = np.sort(self.loc_unmonitored)
        
        
        else:
            if self.h_eps.value.sum() == 0.0 and self.h_zero.value.sum() == 0.0: # placement failure
                order_zero, order_eps = np.zeros(self.n), np.zeros(self.n)
            elif self.p_zero == 0:# no ref.st.
                print('Location for no Reference stations')
                order_zero = []
                order_eps = np.sort(np.argsort(self.h_eps.value)[-self.p_eps:])
            elif self.p_eps == 0:# no LCSs
                print('Location for no LCSs')
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
    
    def compute_Doptimal(self,Psi,alpha):
        """
        Computes D-optimal metric of continuous index obtained from rankMin_reg solution

        Parameters
        ----------
        Psi : numpy array
            low-rank basis.
        alpha : float
            reference station rank regularization parameter

        Returns
        -------
        None.

        """
        precision_matrix = (self.var_eps**-1)*Psi.T@np.diag(self.h_eps.value)@Psi + (self.var_zero**-1)*Psi.T@np.diag(self.h_zero.value)@Psi
        self.Dopt_metric = -1*np.log(np.linalg.det(precision_matrix))
        
        self.logdet_eps = np.log(np.linalg.det(Psi.T@np.diag(self.h_eps.value)@Psi))
        self.trace_zero = alpha*np.trace(Psi.T@np.diag(self.h_zero.value)@Psi)
    
   
        
        
        
# =============================================================================
#         Compute covariance matrix and regressor from solutions
# =============================================================================
                
        
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
        
    def compute_convex_covariance_matrix(self,Psi,weights,var_zero,var_eps,metric='logdet'):
        C_eps = np.diag(weights[0])
        C_refst = np.diag(weights[1])
        Theta_eps = C_eps@Psi
        Theta_refst = C_refst@Psi
        
        Precision_matrix = (var_eps**(-1)*Theta_eps.T@Theta_eps) + (var_zero**(-1)*Theta_refst.T@Theta_refst)
        try:
            self.Cov_convex = np.linalg.inv(Precision_matrix) 
        except:
            print('Computing pseudo-inverse')
            self.Cov_convex = np.linalg.pinv(Precision_matrix)
        
        if metric=='logdet':
            self.metric_convex = np.log(np.linalg.det(self.Cov_convex))
        

    def covariance_matrix(self,Psi,metric='logdet',alpha=0.1,activate_error_solver=True):
        """
        Compute covariance matrix from C matrices and compute a metric

        Parameters
        ----------
        Psi : np array
            low-rank basis used
        metric : str
            metric to compute covariance optimizality: D-optimal, E-optimal, WCS
        alpha: float
            regularization parameter for proposed algorithm

        Returns
        -------
        None.

        """
        C_eps = self.C[0]
        C_zero = self.C[1]
        
        Theta_eps = C_eps@Psi
        Theta_zero = C_zero@Psi
        if self.p_eps !=0:
            self.Precision_matrix = (self.var_eps**(-1)*Theta_eps.T@Theta_eps) + (self.var_zero**(-1)*Theta_zero.T@Theta_zero)
            try:
                self.Cov = np.linalg.inv( self.Precision_matrix ) 
            except:
                print('Computing pseudo-inverse')
                self.Cov = np.linalg.pinv( self.Precision_matrix )
        
        else:
            self.Precision_matrix = (self.var_zero**(-1)*Theta_zero.T@Theta_zero)
            try:
                self.Cov = np.linalg.inv(self.Precision_matrix)
            except:
                print('Computing pseudo-inverse')
                self.Cov = np.linalg.pinv(self.Precision_matrix)
        
        # compute cov-matrix metric
        
        if metric == 'logdet':
            self.metric = np.log(np.linalg.det(self.Cov))
            self.metric_precisionMatrix = np.log(np.linalg.det(self.Precision_matrix))
            
        elif metric == 'eigval':
            self.metric = np.max(np.real(np.linalg.eig(self.Cov)[0]))
            self.metric_precisionMatrix = np.min(np.real(np.linalg.eig(self.Precision_matrix)[0]))
            
        elif metric == 'WCS':
            self.metric = np.diag(self.Cov).max()
        
        elif metric == 'logdet_rank':
            self.metric = np.log(np.linalg.det(self.var_eps**(-1)*Theta_eps.T@Theta_eps)) + alpha*np.trace(self.var_zero**(-1)*Theta_zero.T@Theta_zero)
            
        if activate_error_solver:
            if type(self.problem.value) == type(None):# error when solving
                self.metric = np.inf
                self.metric_precisionMatrix = -np.inf
                
    def covariance_matrix_GLS(self,Psi):
        """
        Compute covariance matrix from GLS. Use pseudo-inverse to account for unstable behavior

        Parameters
        ----------
        Psi : numpy array
            low-rank basis
       
        Returns
        -------
        None.

        """
        C_lcs = self.C[0]
        C_refst = self.C[1]
        Theta_lcs = C_lcs@Psi
        Theta_refst = C_refst@Psi
        
        if C_lcs.shape[0] == 0:# no LCSs
            Precision_matrix = (self.var_zero**-1)*Theta_refst.T@Theta_refst
        elif C_refst.shape[0] == 0:#no Ref.St.
            Precision_matrix = (self.var_eps**-1)*Theta_lcs.T@Theta_lcs
        else:
            Precision_matrix = (self.var_zero**-1)*Theta_refst.T@Theta_refst + (self.var_eps**-1)*Theta_lcs.T@Theta_lcs
        
        try:
            S = np.linalg.svd(Precision_matrix)[1]
            rcond_pinv = rcond_pinv = (S[-1]+S[-2])/(2*S[0])
            Cov = np.linalg.pinv(Precision_matrix,rcond_pinv)
        except:
            Cov = np.linalg.pinv(Precision_matrix,hermitian=True)
      
        self.Cov = Cov
        
    def covariance_matrix_limit(self,Psi):
        """
        Compute covariance matrix in the limit var_zero = 0

        Parameters
        ----------
        Psi : numpy array
            low-rank basis
       

        Returns
        -------
        None.

        """
        C_lcs = self.C[0]
        C_refst = self.C[1]
        Theta_lcs = C_lcs@Psi
        Theta_refst = C_refst@Psi
        
        
        if C_lcs.shape[0] == 0:#no LCSs
            self.Cov = np.zeros(shape=(self.r,self.r))
            return
        elif C_refst.shape[0] == 0:#no RefSt
            Precision_matrix = (self.var_eps**-1)*Theta_lcs.T@Theta_lcs
            S = np.linalg.svd(Precision_matrix)[1]
            rcond_pinv = rcond_pinv = (S[-1]+S[-2])/(2*S[0])
            self.Cov = np.linalg.pinv( Precision_matrix,rcond_pinv)
            
            return
            
        else:
            # compute covariance matrix using projector
            refst_matrix = Theta_refst.T@Theta_refst
            
            Is = np.identity(self.r)
            try:
                P = Is - refst_matrix@np.linalg.pinv(refst_matrix)
            except:
                P = Is - refst_matrix@np.linalg.pinv(refst_matrix,hermitian=True,rcond=1e-10)
            
            rank1 = np.linalg.matrix_rank(Theta_lcs@P,tol=1e-10)
            rank2 = np.linalg.matrix_rank(P@Theta_lcs.T,tol=1e-10)
            
            S1 = np.linalg.svd(Theta_lcs@P)[1]
            S2 = np.linalg.svd(P@Theta_lcs.T)[1]
            
            if rank1==min((Theta_lcs@P).shape):
                try:
                    rcond1_pinv = (S1[-1]+S1[-2])/(2*S1[0])
                except:
                    rcond1_pinv = 1e-15
            else:
                rcond1_pinv = (S1[rank1]+S1[rank1-1])/(2*S1[0])
            
            if rank2==min((P@Theta_lcs.T).shape):
                try:
                    rcond2_pinv = (S2[-1]+S2[-2])/(2*S2[0])
                except:
                    rcond2_pinv = 1e-15
            else:
                rcond2_pinv = (S2[rank2]+S2[rank2-1])/(2*S2[0])
            
            self.Cov = self.var_eps*np.linalg.pinv(Theta_lcs@P,rcond=rcond1_pinv)@np.linalg.pinv(P@Theta_lcs.T,rcond=rcond2_pinv)
         
            
        
        
    def beta_estimated_GLS(self,Psi,y_refst,y_lcs):
        """
        Compute estimated regressor (beta) from sensor measurements

        Parameters
        ----------
        Psi : numpy array
            sparse basis
        y_refst : numpy array
            reference stations vector of measurements
        y_lcs : numpy array
            LCSs vector of measurements

        Returns
        -------
        self.beta_hat : numpy array
                estimated regressor
                Regressor evolution over time (r,num_samples)

        """
        C_eps = self.C[0]
        C_zero = self.C[1]
        Theta_eps = C_eps@Psi
        Theta_zero = C_zero@Psi
        second_term = (self.var_zero**-1)*Theta_zero.T@y_refst + (self.var_eps**-1)*Theta_eps.T@y_lcs
        self.beta_hat = self.Cov@second_term
        
        
    def beta_estimated_limit(self,Psi,y_refst,y_lcs):
        """
        Compute estimated regressor (beta) from sensor measurements
        in the limit variances refst goes to zero (limit of GLS)

        Parameters
        ----------
        Psi : numpy array
            sparse basis
        y_refst : numpy array
            reference stations vector of measurements
        y_lcs : numpy array
            LCSs vector of measurements

        Returns
        -------
        self.beta_hat : numpy array
                estimated regressor over time (r,num_samples)

        """
        C_lcs = self.C[0]
        C_refst = self.C[1]
        Theta_lcs = C_lcs@Psi
        Theta_refst = C_refst@Psi
        refst_matrix = Theta_refst.T@Theta_refst
        
        Is = np.identity(self.r)
        P = Is - refst_matrix@np.linalg.pinv(refst_matrix)
        
        term_refst = np.linalg.pinv(Theta_refst)
        term_lcs = np.linalg.pinv(Theta_lcs@P)@np.linalg.pinv(P@Theta_lcs.T)@Theta_lcs.T
        
        self.beta_hat = term_lcs@y_lcs + term_refst@y_refst
        
        
        
        
        
        
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
    sensor_placement.discretize_solution()
    sensor_placement.C_matrix()
    sensor_placement.covariance_matrix(Psi)