# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:34:34 2023

@author: Agnese
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
import string
from sympy import LeviCivita
from sympy.abc import symbols
from sympy import diff
from sympy import Matrix, matrix_multiply_elementwise

#NUMPY ARRAYS

#%%
"""
Useful functions
"""

def levi_civita_tensor(dim): 
    """
    Compute the Levi-Civita tensor of rank n in
    n-dimensional space.

    Parameters
    ----------
    n: INT
        The dimension of the space.

    Returns
    -------
    TENSOR
        A numpy array representing the Levi-Civita tensor.
        The tensor is a multi-dimensional array of shape (n, n, ..., n).
    """
    arr=np.zeros(tuple([dim for _ in range(dim)]))
    for x in itertools.permutations(tuple(range(dim))):
        mat = np.zeros((dim, dim), dtype=np.int32)
        for i, j in zip(range(dim), x):
            mat[i, j] = 1
        arr[x]=int(np.linalg.det(mat))
    return arr

# def ones(m):
#     """
#     Compute an array of ones of dimension m.

#     Parameters
#     ----------
#     m : INT
#         The dimension of the array.

#     Returns
#     -------
#     TENSOR
        
#     """
#     return [1]*m


# def first_einsum_string(n):
#     """
#     Return a string of the form 'ij,abc->ijabc', to be
#     used in the contraction of LC tensors, when
#     computing the GKP formula.

#     Parameters
#     ----------
#     n : INT
#         Number of transitions.
#     m : INT
#         Number of isotope pairs.

#     Returns
#     -------
#     result_indices : STR

#     """
#     indices1 = ''.join(chr(105 + i) for i in range(n))  # 'ijk...' for n
#     indices2 = ''.join(chr(97 + i) for i in range(n+1))   # 'abc...' for n+1
#     result_indices = indices1 + ',' + indices2 + '->' + indices1 + indices2

#     return result_indices

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
    transition_indices = ''.join(chr(105 + i) for i in range(n))  # 'ijk...' for n
    isotope_pair_indices = ''.join(chr(97 + i) for i in range(n+1))   # 'abc...' for n+1
    fixed_str = transition_indices + ', ' + isotope_pair_indices + ', ' + 'i' + ', ' + 'a' + ', ' + 'b'
    
    
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
Import data in matrix form:
    columns = transitions
    rows = isotope pairs
    --> mass-normalized
"""


#%%

def alpha_plot(mass_vec, alpha_vec):
    """
    Plotting function for alpha_NP.

    Parameters
    ----------
    mass_vec : ARRAY
        contains the mediator mass values.
    alpha_vec : ARRAY
        alpha_NP values corresponding to the values in mass_vec.

    """
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$m_\phi$ (eV)')
    plt.ylabel('$\\alpha_{NP}$')
    plt.plot(mass_vec, alpha_vec)
    
# inches: 
    
#%%


def V_exp_GKP(reduced_data): #_GKP
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

#%%

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
    
    vol = np.einsum(gen_einsum_string(n), levi_civita_tensor(n), levi_civita_tensor(n+1), Xvec, np.ones(n+1), hvec, *matrix_list)
    norm = np.math.factorial(n-1)
    
    res = vol/norm
    
    return res

#NORMALIZATION

#%%
"""
Testing
"""
Xvec = [5, 6]
matrix = np.array([[1/3, 2/4, 3/5], [10/3, 20/4, 30/5]]).T
hvec = [20, 21, 22]

V_th_GKP(Xvec, hvec, matrix)

#%%
Xvec3 = [0.01, 0.02, 0.03]
matrix3 = np.array([[10/5, 20/6, 30/7, 40/5],
                    [50/5, 60/6, 70/7, 80/8],
                    [90/5, 100/6, 110/7, 120/8]]).T
hvec4 = [38, 40, 42, 44]

V_th_GKP(Xvec3, hvec4, matrix3)

#%%
"""
Testing real data
"""

Ca_data = np.array([[2.77187*10**9, 5.34089*10**9, 7.7684*10**9, 9.99038*10**9],
            [-3.5199*10**6,	-6.79247*10**6,	-9.90152*10**6,	-1.27466*10**7],
            [5.39088*10**8,	1.03045*10**9,	1.48114*10**9,	1.8943*10**9]]).T
X_mass1 = [-7.83083*10**15, 3.86395*10**13, 1.56746*10**15]
h = [-1679.21, -1758.78, -1838.21, -1917.75]

V_exp_GKP(Ca_data)/V_th_GKP(X_mass1, h, Ca_data)

#%%

def alpha_GKP(m, elem):
    
    X_mass_elem = np.array(...)[m] #<- row corresponding to mass m
    h_elem = np.array(...)
    data_elem = np.array(...)
    
    alpha = V_exp_GKP(data_elem)/V_th_GKP(X_mass_elem, h_elem, data_elem)
    return alpha

#%%

nT=3
nIP=4

#Define variables to differentiate wrt
nu = symbols(' '.join([f'v{j+1}{i+1}' for j in range(nIP) for i in range(nT)]))
M0 = symbols(' '.join([f'm0{i}' for i in range(1, nIP+1)]))
M = symbols(' '.join([f'm{i}' for i in range(1, nIP+1)]))
A0 = symbols(' '.join([f'A0{i}' for i in range(1, nIP+1)]))
A = symbols(' '.join([f'A{i}' for i in range(1, nIP+1)]))
X = symbols(' '.join([f'X{i}' for i in range(1, nT+1)]))

#Organize them in matrices / vectors
data = Matrix(nIP, nT, nu)
    #mass_pairs = Matrix([[M0[i], M[i]] for i in range(nIP)])
    #A_pairs = Matrix([[A0[i], A[i]] for i in range(nIP)])
AAprime = Matrix([[A0[i] - A[i]] for i in range(nIP)])
red_masses = Matrix([[1/(1/M0[i] - 1/M[i])] for i in range(nIP)])

hvector = matrix_multiply_elementwise(AAprime, red_masses)
reduced_data = Matrix([data.row(a)*red_masses[a] for a in range(nIP)])


#Define placeholders for uncertainites
sig_nu = symbols(' '.join([f'sig_v{j+1}{i+1}' for j in range(nIP) for i in range(nT)]))
sig_M0 = symbols(' '.join([f'sig_m0{i}' for i in range(1, nIP+1)]))
sig_M = symbols(' '.join([f'sig_m{i}' for i in range(1, nIP+1)]))
sig_X = symbols(' '.join([f'sig_X{i}' for i in range(1, nT+1)]))

#%%

# def AAprime(A_pairs):
#     IPpairs = Matrix([])
#     for i in range(len(A_pairs.col(0))):
#         row = sp.Matrix([[A_pairs[i, 0] - A_pairs[i, 1]]])
#         IPpairs = IPpairs.row_insert(i, row)
#     return IPpairs

# def red_masses(mass_pairs):
#     reduced = Matrix([])
#     for i in range(len(mass_pairs.col(0))):
#         row = sp.Matrix([[1/(1/mass_pairs[i, 0] - 1/mass_pairs[i, 1])]])
#         reduced = reduced.row_insert(i, row)
#     return reduced

# mat_data = Matrix([[]])
# for i in range(data.shape[1]):
#     multiplied_column = matrix_multiply_elementwise(data.col(i), red_masses)
#     mat_data = mat_data.col_insert(i, multiplied_column)
#     reduced_data = mat_data.T
    
 

transition_indices = symbols(' '.join([string.ascii_lowercase[8+i] for i in range(0, nT)]))
ip_indices = symbols(' '.join([string.ascii_lowercase[i] for i in range(0, nIP)]))
print(transition_indices)
print(ip_indices)


V_th_sym = 0
for transition_indices in itertools.product(range(nT), repeat=nT):
    for ip_indices in itertools.product(range(nIP), repeat=nIP):
        base = LeviCivita(*transition_indices)*LeviCivita(*ip_indices)*X[transition_indices[0]]*hvector[ip_indices[1]]
        if base != 0:
            print('base = ', base)
            test = reduced_data.col(transition_indices[1])[ip_indices[2]]
            print('test = ', test)
            res = base*test
            print('res = ', res)
        #V_th_sym += LeviCivita(*transition_indices)*LeviCivita(*ip_indices)*X[transition_indices[0]]*hvector[ip_indices[1]]*reduced_data.col(transition_indices[1])[ip_indices[2]]*reduced_data.col(transition_indices[2])[ip_indices[3]]

print(res)


#%%

  

V_th_sym = 0
for i, j, k in itertools.product(range(3), repeat=3):
    for a, b, c, d in itertools.product(range(4), repeat=4):
        V_th_sym += LeviCivita(i,j,k)*LeviCivita(a,b,c,d)*X[i]*hvector[b]*reduced_data.col(j)[c]*reduced_data.col(k)[d]

print(V_th_sym)

# derivatives = 0
# for i in range(len(nu)):
#     derivatives += diff(V_th_sym, nu[i])
# for i in range(len(M)):
#     derivatives += diff(V_th_sym, M[i])
#     for i in range(len(M0)):
#         derivatives += diff(V_th_sym, M0[i])
# for i in range(len(X)):
#     derivatives += diff(V_th_sym, X[i])
    
#%%


import string


def generate_iterables(num_loops):
    letters = string.ascii_lowercase[:num_loops]
    return itertools.combinations_with_replacement(letters, num_loops)

n = 3  # Number of nested loops

iterables = generate_iterables(n)

for index_set in iterables:
    print(index_set)
    
    

#%%
# We can use expr.subs(x, 2) to assign the value 2 to the symbol x in expr.

# We can evaluate the expression usign eval() or evalf() (for floating
# points); it's possible to specify the numer of digit as argument.

# To numerically evaluate an expression w/ a symbol @ a point:
# expr = cos(2*x)
# expr.evalf(subs={x: 2.4})

# lambify converts SymPy to other libraries

i, j, k = symbols('i j k')
a, b, c, d = symbols('a b c d')

expr = 0
for i, j, k in itertools.product(range(3), repeat=3):
    for a, b, c, d in itertools.product(range(4), repeat=4):
        base = LeviCivita(i,j,k)*LeviCivita(a,b,c,d)*X[i]*hvector[b]
        for i in range(nT-1):
            base *= reduced_data[indices[i+1]][c]
        #print(terms)
        summation = sum(terms)

print(summation)

#%%


V_th_sym = 0
for i, j in itertools.product(range(2), repeat=2):
    for a, b, c in itertools.product(range(3), repeat=4):
        V_th_sym += LeviCivita(i,j)*LeviCivita(a,b,c)*X[i]*hvector[b]*reduced_data.col(j)[c]

print(V_th_sym)


#%%


reduced_data.col(1)[c]








