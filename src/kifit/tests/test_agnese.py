# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 17:42:42 2023

@author: Agnese
"""

import numpy as np
import sympy as sp

from builder import Element
from Mathematica_outputs import *

def test_volume_data_gkp():
    ca = Element('Ca_KP')
    
    assert len(ca.mu_norm_isotope_shifts) == 3
    m = len(ca.mu_norm_isotope_shifts)
    assert m == 3

    assert len(ca.mu_norm_isotope_shifts.T) == 2
    
    datamatrix_extended = np.hstack((ca.mu_norm_isotope_shifts,
                                      np.ones(m)[:, np.newaxis]))

    np.allclose(datamatrix_extended, mu_norm_isotope_shifts_Mathematica, atol=0.01)
    
    np.isclose(ca.volume_data_gkp, volume_data_gkp_Mathematica, atol=0.01)

def test_volume_theory_gkp():
    ca = Element('Ca_KP')
    
    np.isclose(ca.volume_theory_gkp, volume_theory_gkp_Mathematica, atol=0.01)
    
def test_alpha_gkp():
    ca = Element('Ca_KP')
    
    np.isclose(ca.alpha_gkp, alpha_gkp_Mathematica, atol=0.01)

def test_alpha_gkp_symbolic():
    ca = Element('Ca_KP')
    
    assert sp.Eq(ca.alpha_gkp_symbolic, alpha_gkp_symbolic_Mathematica)

ca = Element('Ca_KP')
ca.alpha_gkp_symbolic