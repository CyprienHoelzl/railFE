# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:21:46 2020

# Analytic Rail beam formulation
  For the rail formulation a timoshenko 4 beam

@author: CyprienHoelzl
"""
import numpy as np
#%% Symbolic analytic shape functions - Matlab is much more powerfull here
import sympy as sym
# sym.init_printing()
alpha,beta,gamma, A1,A2,A3,A4, x = sym.symbols('alpha beta gamma A1 A2 A3 A4 x', real=True)
kdEIkAg2, EIkAG, L = sym.symbols('kdEIkAg2 EIkAG L',real=True)
sym.Function('w')

# Displacement function
w = A1*sym.exp(gamma*x) +A2*x*sym.exp(gamma*x) + A3*sym.exp(beta*x)*sym.sin(alpha*x) + A4*sym.exp(beta*x)*sym.cos(alpha*x)

w_prim = sym.diff(w,x)
w_prim3= sym.diff(w,x,3)
# Theta function
theta = w_prim*(1-kdEIkAg2)+EIkAG*w_prim3

# g coefficients at boundaries:
gr = np.eye(4)
g_coeffs = sym.Matrix.zeros(4,4)

for i in range(4):
    g0 = w.subs([(A1,gr[0,i]),(A2,gr[1,i]),(A3,gr[2,i]),(A4,gr[3,i]),(x,0)])
    h0 = theta.subs([(A1,gr[0,i]),(A2,gr[1,i]),(A3,gr[2,i]),(A4,gr[3,i]),(x,0)])
    gL = w.subs([(A1,gr[0,i]),(A2,gr[1,i]),(A3,gr[2,i]),(A4,gr[3,i]),(x,L)])
    hL = theta.subs([(A1,gr[0,i]),(A2,gr[1,i]),(A3,gr[2,i]),(A4,gr[3,i]),(x,L)])
    g_coeffs[0,i] = g0
    g_coeffs[1,i] = h0
    g_coeffs[2,i] = gL
    g_coeffs[3,i] = hL
g_coeff_s = sym.simplify(g_coeffs)
Coeffs = g_coeff_s.LUsolve(sym.eye(4))
N0 = w.subs([(A1,Coeffs[0,0]),(A2,Coeffs[1,0]),(A3,Coeffs[2,0]),(A4,Coeffs[3,0])])
N0 = w.subs([(A1,Coeffs[0,1]),(A2,Coeffs[1,1]),(A3,Coeffs[2,1]),(A4,Coeffs[3,1])])
N0 = w.subs([(A1,Coeffs[0,2]),(A2,Coeffs[1,2]),(A3,Coeffs[2,2]),(A4,Coeffs[3,2])])
N0 = w.subs([(A1,Coeffs[0,3]),(A2,Coeffs[1,3]),(A3,Coeffs[2,3]),(A4,Coeffs[3,3])])
