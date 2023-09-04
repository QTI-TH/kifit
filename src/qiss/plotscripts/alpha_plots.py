# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:34:34 2023

@author: Agnese
"""

import numpy as np
import sympy as sp
import itertools
import matplotlib.pyplot as plt
import string
from sympy import LeviCivita
from sympy.abc import symbols
from sympy import diff
from sympy import Matrix, matrix_multiply_elementwise

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

def V_exp_GKP(reduced_data):
    """
    Compute volume of NLs from data for GKP formula.
    
    Parameters
    ----------
    reduced_data : ARR
        Contains reduced isotope shifts (nxm);
        columns (n) = reduced isotope shifts;
        rows (m) = isotope pairs.
        

    Returns
    -------
    INT
        Computes determinant of reduced_data.

    """
    m = len(reduced_data)
    if m != len(reduced_data.T) + 1:
        print('Wrong shape of data matrix')
        
    datamatrix_extended = np.hstack((reduced_data, np.ones(m)[:, np.newaxis]))
    
    return np.linalg.det(datamatrix_extended)

def V_th_GKP(Xvec, hvec, reduced_data):
    """
    Compute theory volume of NLs for the GKP formula, for alpha_NP=1.

    Parameters
    ----------
    Xvec : ARR
        Vector containing the X coefficients of the transitions
        for fixed mediator mass.
    hvec : ARR
        Vector containing h=gamma/mu.
    datamatrix : ARR
        nxm matrix containing reduced isotope shifts.

    Returns
    -------
    res : INT
        DESCRIPTION.

    """
    if len(Xvec) != len(reduced_data.T):
        print('Wrong dimension of X vector')
    
    if len(hvec) != len(reduced_data):
        print('Wrong dimension of h vector')
        
    n = len(Xvec)
    matrix_list = [reduced_data] * (n-1)
    
    vol = np.einsum(gen_einsum_string(n), levi_civita_tensor(n),
                    levi_civita_tensor(n+1), Xvec, np.ones(n+1), hvec,
                    *matrix_list)
    norm = np.math.factorial(n-1)
    
    Vtheo = vol/norm
    
    return Vtheo

def alpha_GKP(mass): #
    
    X_mass = np.array(...)[mass] #<-row corresponding to mass m
    h = np.array(...)
    data_matrix = np.array(...)
    
    alpha = V_exp_GKP(data_matrix)/V_th_GKP(X_mass, h, data_matrix)
    return alpha


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

#%%

"""
Plotting functions
To be moved to a different script
"""

def alpha_mass_plot(mass_vec, alpha_vec):
    """
    Plotting function for alpha_NP.

    Parameters
    ----------
    mass_vec : ARRAY
        contains the mediator mass values.
    alpha_vec : ARRAY
        alpha_NP values corresponding to the values in mass_vec;
        computed in the Elem class

    """
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$m_\phi$ (eV)')
    plt.ylabel('$\\alpha_{NP}$')
    # change plot size
    plt.plot(mass_vec, alpha_vec)
    






