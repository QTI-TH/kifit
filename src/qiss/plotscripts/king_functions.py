# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 14:33:55 2023

@author: Agnese
"""
import numpy as np
import sympy as sp
import itertools
import string
from sympy import LeviCivita
from sympy.abc import symbols
from sympy import diff
from sympy import Matrix, matrix_multiply_elementwise

import loadelems

#%%

"""
Useful functions
"""

def levi_civita_tensor(dim): 
    """
    Compute the Levi-Civita tensor of rank dim in
    dim-dimensional space.

    Parameters
    ----------
    dim: INT
        The dimension of the space.

    Returns
    -------
    TENSOR
        A numpy array representing the Levi-Civita tensor.
        The tensor is a multi-dimensional array of shape (dim, dim, ..., dim).
    """
    arr=np.zeros(tuple([dim for _ in range(dim)]))
    for x in itertools.permutations(tuple(range(dim))):
        mat = np.zeros((dim, dim), dtype=np.int32)
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x]=int(np.linalg.det(mat))
    return arr

def gen_einsum_string(n):
    """
    Generate a string of the form 'ij, abc, i, a, b, cj', to be
    used in the Einstein summation for the GKP formula.

    Parameters
    ----------
    n : INT
        Number of transitions.

    Returns
    -------
    einsum_str : STR

    """
    col_row_str = [] 
    transition_indices = ''.join(chr(105 + i) for i in range(n))
                # 'ijk...' for n
    isotope_pair_indices = ''.join(chr(97 + i) for i in range(n+1))
                # 'abc...' for n+1
    fixed_str = (transition_indices + ', ' + isotope_pair_indices 
                 + ', ' + 'i' + ', ' + 'a' + ', ' + 'b')
    
    
    for i in range(n-1):
        row_index = ''.join(chr(99 + i)) #'c'... for n-1
        col_index = ''.join(chr(106 + i)) #'j'... for n-1
        tot = row_index + col_index #'cj'
        col_row_str.append(tot) #'cj, dk,...' for n-1
    
    mat_indices_str = ', '.join(col_row_str)
    
    einsum_str = fixed_str + ', ' + mat_indices_str
    
    return einsum_str

#%%  
    
"""
King Plot functions
"""

# @cached_fct_property
def V_exp_GKP(self):
    """
    Return volume of NLs in mass-normalised isotope shifts for GKP formula.
    
        Add an m-dimensional identity column to mass-normalised data matrix
        and calculate the determinant of the extended matrix.
        
    """
    
    m = len(self.mu_norm_isotope_shifts)
    if m != len(self.mu_norm_isotope_shifts.T) + 1:
        print('Wrong shape of data matrix')
        
    datamatrix_extended = np.hstack((self.mu_norm_isotope_shifts,
                                     np.ones(m)[:, np.newaxis]))
    
    return np.linalg.det(datamatrix_extended)

def V_th_GKP(self, x: int):
    """
    Return theory volume of NLs for the GKP formula, for alpha_NP=1
    for a given mediator mass

    """
    self.X = self.Xcoeffs[x, 1:]
    
    if len(self.X) != len(self.mu_norm_isotope_shifts.T):
        print('Wrong dimension of X vector')
    
    if len(self.h_aap) != len(self.mu_norm_isotope_shifts):
        print('Wrong dimension of h vector')
    # These checks are already done when building the king, does it make
    # sense to do them again?
        
    n = len(self.X)
    matrix_list = [self.mu_norm_isotope_shifts] * (n-1)
    
    vol = np.einsum(gen_einsum_string(n), levi_civita_tensor(n),
                    levi_civita_tensor(n+1), self.X, np.ones(n+1), self.h_aap,
                    *matrix_list)
    norm = np.math.factorial(n-1)
    
    return vol/norm

def alpha_GKP(self, x: int): 
    """
    Return alpha_NP for the GKP formula for a given mediator mass.

    """
    return V_exp_GKP(self)/V_th_GKP(self, x)


def V_exp_GKP_symbolic(nT, nIP):

    nT = 2
    nIP = 3
    
    #Define variables
    nu = symbols(' '.join([f'v{j+1}{i+1}' for j in range(nIP)
                           for i in range(nT)]))
    M0 = symbols(' '.join([f'm0{i}' for i in range(1, nIP+1)]))
    M = symbols(' '.join([f'm{i}' for i in range(1, nIP+1)]))
    
    #Organize variables in matrices / vectors
    data = Matrix(nIP, nT, nu)
    red_masses = Matrix([[1/(1/M0[i] - 1/M[i])] for i in range(nIP)])
    reduced_data = Matrix([data.row(a)*red_masses[a] for a in range(nIP)])
    
    GKP_square_data = reduced_data.col_insert(nT, Matrix(np.ones(nIP)))
    V_exp_sym = GKP_square_data.det()
    
    
    return V_exp_sym


def V_th_GKP_symbolic(nT, nIP):
    """
    

    Parameters
    ----------
    nT : INT
        Number of transitions.
    nIP : INT
        Number of isotope pairs.

    Returns
    -------
    V_th_sym : SYM
        Symbolic form of the theory volume for GKP.

    """
    
    #Define variables
    nu = symbols(' '.join([f'v{j+1}{i+1}' for j in range(nIP)
                           for i in range(nT)]))
    M0 = symbols(' '.join([f'm0{i}' for i in range(1, nIP+1)]))
    M = symbols(' '.join([f'm{i}' for i in range(1, nIP+1)]))
    A0 = symbols(' '.join([f'A0{i}' for i in range(1, nIP+1)]))
    A = symbols(' '.join([f'A{i}' for i in range(1, nIP+1)]))
    X = symbols(' '.join([f'X{i}' for i in range(1, nT+1)]))

    #Organize variables in matrices / vectors
    data = Matrix(nIP, nT, nu)
    AAprime = Matrix([[A0[i] - A[i]] for i in range(nIP)])
    red_masses = Matrix([[1/(1/M0[i] - 1/M[i])] for i in range(nIP)])
    hvector = matrix_multiply_elementwise(AAprime, red_masses)
    reduced_data = Matrix([data.row(a)*red_masses[a] for a in range(nIP)])

    #Define indices for transitions and isotope pairs
    transition_indices = symbols(' '.join([string.ascii_lowercase[8+i]
                                           for i in range(0, nT)]))
    ip_indices = symbols(' '.join([string.ascii_lowercase[i]
                                   for i in range(0, nIP)]))

    # Build volume
    V_th_sym = 0
    for transition_indices in itertools.product(range(nT), repeat=nT):
        for ip_indices in itertools.product(range(nIP), repeat=nIP):
            base = (LeviCivita(*transition_indices)*LeviCivita(*ip_indices)
                    *X[transition_indices[0]]*hvector[ip_indices[1]])
            for w in range(1, nT):
                add = reduced_data.col(transition_indices[w])[ip_indices[w+1]]
                base *= add
                V_th_sym += base


    return V_th_sym


def alpha_GKP_symbolic(nT, nIP):
    
    alpha_sym = V_exp_GKP_symbolic(nT, nIP)/V_th_GKP_symbolic(nT, nIP)
    
    return alpha_sym

def sig_alpha_symbolic(nT, nIP):
    
    #Define symbolic variables
    nu = symbols(' '.join([f'v{j+1}{i+1}' for j in range(nIP)
                           for i in range(nT)]))
    M0 = symbols(' '.join([f'm0{i}' for i in range(1, nIP+1)]))
    M = symbols(' '.join([f'm{i}' for i in range(1, nIP+1)]))
    X = symbols(' '.join([f'X{i}' for i in range(1, nT+1)]))
    
    #Define placeholders for uncertainites
    sig_nu = symbols(' '.join([f'sig_v{j+1}{i+1}'
                               for j in range(nIP) for i in range(nT)]))
    sig_M0 = symbols(' '.join([f'sig_m0{i}' for i in range(1, nIP+1)]))
    sig_M = symbols(' '.join([f'sig_m{i}' for i in range(1, nIP+1)]))
    sig_X = symbols(' '.join([f'sig_X{i}' for i in range(1, nT+1)]))
    
    
    derivatives = 0
    for i in range(len(nu)):
        derivatives += diff(alpha_GKP_symbolic(nT, nIP), nu[i])**2*sig_nu[i]**2
    for i in range(len(M)):
        derivatives += diff(alpha_GKP_symbolic(nT, nIP), M[i])**2*sig_M[i]**2
        for i in range(len(M0)):
            derivatives += (diff(alpha_GKP_symbolic(nT, nIP), M0[i])**2
                            *sig_M0[i]**2)
    for i in range(len(X)):
        derivatives += diff(alpha_GKP_symbolic(nT, nIP), X[i])**2*sig_X[i]**2
    
    sig_alpha_th_sym = sp.sqrt(derivatives)
    
    return sig_alpha_th_sym