# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 18:34:34 2023

@author: Agnese
"""

import matplotlib.pyplot as plt


"""
Plotting functions
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
    






