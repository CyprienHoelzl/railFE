# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:21:46 2020

Unit Conversions  

@author: Cyprien Hoelzl
"""
import numpy as np
#%% Unit conversions
def P2R(radii, angles):
    """
    Polar to rectangular

    Parameters
    ----------
    radii : radius.
    angles : angle in rad.

    Returns
    -------
    rectangular coordinates (a+b*j)

    """
    
    return radii * np.exp(1j*angles)

def R2P(x):
    """
    Rectangular to polar

    Parameters
    ----------
    x : rectangular coordinates (a+b*j)

    Returns
    -------
    radii : radius
    angles : angles

    """
    radii, angles = np.abs(x), np.angle(x)
    return radii,angles
