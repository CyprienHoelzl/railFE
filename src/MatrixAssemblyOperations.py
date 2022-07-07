# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:21:46 2020

MatrixAssemblyOperations    

@author: CyprienHoelzl
"""
import numpy as np
#%% Matrix Assembly Operations
def addAtPos(mat1, mat2, xypos):
    """
    Add two arrays of different sizes in place, offset by xy coordinates

    Parameters
    ----------
    mat1 : base array
    mat2 : secondary array added to mat1
    xypos : tuple (x,y) containing coordinates

    Returns
    -------
    mat1 : array1 + (array2 added at position (x,y) in array1)

    """
    x, y = xypos
    ysize, xsize = mat2.shape
    xmax, ymax = (x + xsize), (y + ysize)
    mat1[y:ymax, x:xmax] += mat2
    return mat1

def addAtIdx(mat1, mat2, xypos):
    """
    Add two arrays of different sizes in place, offset by xy coordinates
    

    Parameters
    ----------
    mat1 : array1 of size (a,b)
    mat2 : array2 of size (n,m) to be added to mat1
    xypos : tuple ([x,x2,x3...xn],[y1,y2,y3,..ym]) containing coordinates in mat1 for elements in mat2

    Returns
    -------
    mat1 : summed arrays
    
    """
    x, y = xypos
    mat1[np.ix_(y, x)] += mat2
    return mat1