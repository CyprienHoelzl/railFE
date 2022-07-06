# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:21:46 2020

MatrixAssemblyOperations    

@author: CyprienHoelzl
"""
#%% Matrix Assembly Operations
def addAtPos(mat1, mat2, xypos):
    """
    Add two matrices of different sizes in place, offset by xy coordinates
    Usage:
      - mat1: base matrix
      - mat2: add this matrix to mat1
      - xypos: tuple (x,y) containing coordinates
    """
    x, y = xypos
    ysize, xsize = mat2.shape
    xmax, ymax = (x + xsize), (y + ysize)
    mat1[y:ymax, x:xmax] += mat2
    return mat1

def addAtIdx(mat1, mat2, xypos):
    """
    Add two matrices of different sizes in place, offset by xy coordinates
    Usage:
      - mat1: base matrix
      - mat2: add this matrix to mat1
      - xypos: tuple ([x,x2,x3...],[y1,y2,y3,..]) containing coordinates in mat1
    """
    x, y = xypos
    mat1[np.ix_(y, x)] += mat2
    return mat1

    
def is_odd(self,num):
    return num & 0x1