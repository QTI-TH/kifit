# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 14:33:55 2023

@author: Agnese
"""
import itertools

import numpy as np
import sympy as sp
from sympy import LeviCivita
from sympy.abc import symbols
from sympy import diff
from sympy import Matrix, matrix_multiply_elementwise

# import builder

#%%

"""
Useful functions
"""

def levi_civita_tensor(d): 
    """
    Return the Levi-Civita tensor as a d-dimensional array

    """
    arr=np.zeros([d for _ in range(d)])
    for x in itertools.permutations(tuple(range(d))):
        mat = np.zeros((d, d), dtype=np.int32)
        for i, j in enumerate(x):
            mat[i][j] = 1
        arr[x]=int(np.linalg.det(mat))
    return arr

def generate_einsum_string(n):
    """
    Return a string of the form 'ij, abc, i, a, b, cj, dk, ...'
    where the stopping point is determined by n.

    """
     
    fixed_transition_indices = ''.join(chr(105 + i) for i in range(n))
                # 'ijk...' for n
    fixed_isotope_pair_indices = ''.join(chr(97 + i) for i in range(n+1))
                # 'abc...' for n+1
    fixed_string = (fixed_transition_indices + ', '
                    + fixed_isotope_pair_indices + ', '
                    + 'i' + ', ' + 'a' + ', ' + 'b')
        
    dynamic_string = []
    for i in range(n-1):
        row_indices = ''.join(chr(99 + i)) #'c'... for n-1
        column_indices = ''.join(chr(106 + i)) #'j'... for n-1
        combi_string = row_indices + column_indices #'cj'
        dynamic_string.append(combi_string) #'cj, dk,...' for n-1
    
    matrix_indices_string = ', '.join(dynamic_string)
    
    einsum_string = fixed_string + ', ' + matrix_indices_string
    
    return einsum_string

#%%  
    
"""
King Plot functions
"""

# @cached_fct_property
def volume_data_gkp(self):
    """
    Return volume of NLs in mass-normalised isotope shifts for GKP formula.
    
        Add an m-dimensional identity column to mass-normalised data matrix
        and calculate the determinant of the extended matrix.
        
    """
    
    m = len(self.mu_norm_isotope_shifts)
    if m != len(self.mu_norm_isotope_shifts.T) + 1:
        raise ValueError('Wrong shape of data matrix')
        
    datamatrix_extended = np.hstack((self.mu_norm_isotope_shifts,
                                     np.ones(m)[:, np.newaxis]))
    
    return np.linalg.det(datamatrix_extended)

# @cached_fct_property
def volume_theory_gkp(self, x: int):
    """
    Return theory volume of NLs for the GKP formula, for alpha_NP=1
    for a given mediator mass

    """
    self.X = self.Xcoeffs[x, 1:]
    
    if len(self.X) != len(self.mu_norm_isotope_shifts.T):
        raise ValueError('Wrong dimension of X vector')
    
    if len(self.h_aap) != len(self.mu_norm_isotope_shifts):
        raise ValueError('Wrong dimension of h vector')
    # These checks are already done when building the king, does it make
    # sense to do them again?
        
    n = len(self.X)
    matrix_list = [self.mu_norm_isotope_shifts] * (n-1)
    
    vol = np.einsum(generate_einsum_string(n), levi_civita_tensor(n),
                    levi_civita_tensor(n+1), self.X, np.ones(n+1), self.h_aap,
                    *matrix_list)
    norm = np.math.factorial(n-1)
    
    return vol/norm

# @cached_fct_property
def alpha_gkp(self, x: int): 
    """
    Return alpha_NP for the GKP formula for a given mediator mass.

    """
    volume_fixed_mphi = volume_theory_gkp(self, x)
    
    return self.volume_data_gkp/volume_fixed_mphi

# @cached_fct_property
def volume_data_gkp_symbolic(self):
    """
    Return symbolic expression of NLs volume from data for the GKP formula.
    
    """

    nIP = self.m_nisotopepairs
    nT = self.n_ntransitions
    
    #Define variables
    nu = symbols(' '.join([f'v{j+1}{i+1}' for j in range(nIP)
                           for i in range(nT)]))
    m = symbols(' '.join([f'm0{i}' for i in range(1, nIP+1)]))
    mp = symbols(' '.join([f'm{i}' for i in range(1, nIP+1)]))
    
    #Organize variables in matrices / vectors
    data = Matrix(nIP, nT, nu)
    red_masses = Matrix([[1/(1/m[i] - 1/mp[i])] for i in range(nIP)])
    reduced_data = Matrix([data.row(a)*red_masses[a] for a in range(nIP)])
    
    gkp_square_data = reduced_data.col_insert(nT, Matrix(np.ones(nIP)))
    
    return gkp_square_data.det()

# @cached_fct_property
def volume_theory_gkp_symbolic(self):
    """
    Return symbolic expression of NLs volume from theory for the GKP formula.
    
    """
    n_isotope_pairs = self.m_nisotopepairs
    n_transitions = self.n_ntransitions
    
    # Define symbolic variables
    nu = symbols(' '.join([f'v{j+1}{i+1}' for j in range(n_isotope_pairs)
                           for i in range(n_transitions)]))
    m = symbols(' '.join([f'm0{i}' for i in range(1, n_isotope_pairs+1)]))
    mp = symbols(' '.join([f'm{i}' for i in range(1, n_isotope_pairs+1)]))
    a = symbols(' '.join([f'A0{i}' for i in range(1, n_isotope_pairs+1)]))
    ap = symbols(' '.join([f'A{i}' for i in range(1, n_isotope_pairs+1)]))
    x = symbols(' '.join([f'X{i}' for i in range(1, n_transitions+1)]))

    # Organize symbolic variables in matrices and vectors
    data = Matrix(n_isotope_pairs, n_transitions, nu)
    aap = Matrix([[a[i] - ap[i]] for i in range(n_isotope_pairs)])
    red_masses = Matrix([[1/(1/m[i] - 1/mp[i])]
                         for i in range(n_isotope_pairs)])
    hvector = matrix_multiply_elementwise(aap, red_masses)
    reduced_data = Matrix([data.row(a)*red_masses[a]
                           for a in range(n_isotope_pairs)])

    # Define indices for transitions and isotope pairs
    transition_indices = symbols(' '.join([chr(8+1)
                                           for i in range(0, n_transitions)]))
    ip_indices = symbols(' '.join([chr(i) for i in range(0, n_isotope_pairs)]))

    # Build symbolic expression of NLs volume
    vol_th_sym = 0
    for transition_indices in itertools.product(range(n_transitions),
                                                repeat=n_transitions):
        for ip_indices in itertools.product(range(n_isotope_pairs),
                                            repeat=n_isotope_pairs):
            base = (LeviCivita(*transition_indices)*LeviCivita(*ip_indices)
                    *x[transition_indices[0]]*hvector[ip_indices[1]])
            for w in range(1, n_transitions):
                add = reduced_data.col(transition_indices[w])[ip_indices[w+1]]
                base *= add
                vol_th_sym += base


    return vol_th_sym


def alpha_gkp_symbolic(self):
    """
    Return symbolic expression of alpha_NP for the GKP formula.
    
    """
    return self.volume_data_gkp_symbolic/self.volume_theory_gkp_symbolic

def sig_alpha_gkp_symbolic(self):
    """
    Return symbolic expression of alpha_NP for the GKP formula.
    
    """
    n_isotope_pairs = self.m_nisotopepairs
    n_transitions = self.n_ntransitions
    
    #Define symbolic variables
    nu = symbols(' '.join([f'v{j+1}{i+1}' for j in range(n_isotope_pairs)
                           for i in range(n_transitions)]))
    m = symbols(' '.join([f'm0{i}' for i in range(1, n_isotope_pairs+1)]))
    mp = symbols(' '.join([f'm{i}' for i in range(1, n_isotope_pairs+1)]))
    x = symbols(' '.join([f'X{i}' for i in range(1, n_transitions+1)]))
    
    #Define placeholders for uncertainites
    sig_nu = symbols(' '.join([f'sig_v{j+1}{i+1}'
                               for j in range(n_transitions)
                               for i in range(n_transitions)]))
    sig_m = symbols(' '.join([f'sig_m0{i}'
                               for i in range(1, n_isotope_pairs+1)]))
    sig_mp = symbols(' '.join([f'sig_m{i}'
                              for i in range(1, n_isotope_pairs+1)]))
    sig_x = symbols(' '.join([f'sig_X{i}' for i in range(1, n_transitions+1)]))
    
    
    derivatives = 0
    for i in range(len(nu)):
        derivatives += diff(alpha_gkp_symbolic(n_transitions, n_transitions),
                            nu[i])**2*sig_nu[i]**2
    for i in range(len(mp)):
        derivatives += diff(alpha_gkp_symbolic(n_transitions, n_transitions),
                            mp[i])**2*sig_mp[i]**2
        for i in range(len(m)):
            derivatives += (diff(alpha_gkp_symbolic(n_transitions,
                                                    n_transitions), m[i])**2
                            *sig_m[i]**2)
    for i in range(len(x)):
        derivatives += diff(alpha_gkp_symbolic(n_transitions,n_transitions),
                            x[i])**2*sig_x[i]**2
    
    return sp.sqrt(derivatives)
