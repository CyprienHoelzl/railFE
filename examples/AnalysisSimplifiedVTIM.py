# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:21:46 2020

Simplified Vehicle-Track interaction model
@author: e512481
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.linalg import pascal
from scipy.sparse import linalg
import scipy.integrate as integrate
import scipy

# module imports
from VehicleModelAssembly import VehicleAssembly
from TrackModelAssembly import UIC60properties,Pad
from TimoshenkoBeamModel import Timoshenko4,Timoshenko4eb
from SystemAssembly import OverallSystem

#%% Unit conversions
def P2R(radii, angles):
    return radii * np.exp(1j*angles)

def R2P(x):
    return np.abs(x), np.angle(x)

#%% Plotting       
def plot_shapefunctions():
    UIC60props = UIC60properties()
    tim4 = Timoshenko4(UIC60props,0.3)
    fig,ax = plt.subplots(figsize=(4,3))
    ax.plot(tim4.N_w1(np.arange(0,1.1,0.1)),label = r'$N_{w_1}(\xi)$')
    ax.plot(tim4.N_t1(np.arange(0,1.1,0.1)),label = r'$N_{\theta_1}(\xi)$')
    ax.plot(tim4.N_w2(np.arange(0,1.1,0.1)),label = r'$N_{w_2}(\xi)$')
    ax.plot(tim4.N_t2(np.arange(0,1.1,0.1)),label = r'$N_{\theta_2}(\xi)$')
    ax.legend()
    ax.set_xlabel(r"$\xi = x/L$")
    ax.set_ylabel('$[-]$')
    ax.set_title('Interpolation Functions for TIM4')
    return fig,ax

def plot_shapefunctions_el():
    UIC60props = UIC60properties()
    tim4el = Timoshenko4eb(UIC60props,Pad(),0.14)
    
    fig,ax = plt.subplots(figsize=(4,3))
    ax.plot(tim4el.N_w1(np.arange(0,1.1,0.1)),label = r'$N_{w_1}(\xi)$')
    ax.plot(tim4el.N_t1(np.arange(0,1.1,0.1)),label = r'$N_{\theta_1}(\xi)$')
    ax.plot(tim4el.N_w2(np.arange(0,1.1,0.1)),label = r'$N_{w_2}(\xi)$')
    ax.plot(tim4el.N_t2(np.arange(0,1.1,0.1)),label = r'$N_{\theta_2}(\xi)$')
    ax.plot(tim4el.N_s(np.arange(0,1.1,0.1)),label = r'$N_{s}(\xi)$')
    ax.legend()
    ax.set_xlabel(r"$\xi = x/L$")
    ax.set_ylabel('$[-]$')
    ax.set_title('Interpolation Functions for TIM4 elastic bedding')
    return fig,ax
def plot_comparison():
    fig,ax = plt.subplots(figsize=(7,3),ncols=2)
    fig.suptitle('TIM4 difference for distributed stiffness')
    for i in np.arange(200,1200,200):
        UIC60props = UIC60properties()
        tim4 = Timoshenko4(UIC60props,0.5)
        tim4el = Timoshenko4eb(UIC60props,Pad(K_p=i*10**6),0.5)
        ax[0].plot(-tim4el.N_w1(np.arange(0,1.1,0.1))+tim4.N_w1(np.arange(0,1.1,0.1)),label = '${} [MN/m^2]$'.format(str(i)))#r'$N_{w1}(\xi)$')
        ax[1].plot(-tim4el.N_w2(np.arange(0,1.1,0.1))+tim4.N_w2(np.arange(0,1.1,0.1)),label = '${} [MN/m^2]$'.format(str(i)))#r'$N_{w2}(\xi)$')
        ax[0].legend()
        ax[0].set_xlabel(r"$\xi = x/L$")
        ax[1].set_xlabel(r"$\xi = x/L$")
        ax[0].set_ylabel('$N_{1}-N_{1.k_p} [-]$')
        ax[1].set_ylabel('$N_2-N_{2.k_p} [-]$')
    fig.subplots_adjust(wspace=0.28)
    fig.tight_layout()
    return fig,ax  
def plot_sleeperstiffnessNs():
    from mpl_toolkits.mplot3d import Axes3D 
    from matplotlib import cm
    X = np.arange(0,1.01,0.01)
    Y = np.arange(10,800,10)
    Z = []
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # fig.suptitle('TIM4 difference for distributed stiffness')
    for i in Y:
        UIC60props = UIC60properties()
        tim4el = Timoshenko4eb(UIC60props,Pad(K_p=i*10**6),0.5)
        Z.append(tim4el.N_s(X))
    Z = np.vstack(Z)
    # Plot the surface.
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Customize plot
    ax.set_xlabel(r"$\xi = x/L$")
    ax.set_ylabel('Stiffness $[MN/m]$')
    ax.set_zlabel('$N_S [-]$')
    
    ax.set_xlim([0,1])
    ax.set_ylim([0,800])
    ax.set_zlim([0,1])
    ax.set_title('Interpolation function for the sleeper displacement for flexible track element')
    return fig,ax  
#%% Application
plot=True#False
if plot==True:
    fig,ax = plot_shapefunctions()
    fig,ax = plot_shapefunctions_el()
    fig,ax = plot_comparison()
    fig,ax = plot_sleeperstiffnessNs()

UIC60props = UIC60properties()
tim4 = Timoshenko4(UIC60props,0.16)

K_e = tim4.stiffnessMatrix()
M_e = tim4.massMatrix()

tim4el = Timoshenko4eb(UIC60props,Pad(),0.16)
S_k_e,S_c_e= tim4el.kc_contribution_bedding()
K_ef = tim4el.stiffnessMatrixTEEF()
D_ef = tim4el.dampingMatrixTEEF()
M_ef = tim4el.massMatrixTEEF()

w_ef = tim4el.particular_solution_sleeper()


#%% Frequency Domain Solution
# H = (-w^2*M + iwC+K)**-1
# or x'= Ax+Bu
#    y = Cx+Du=0
OverallSyst = OverallSystem(xi=0.5)
M = OverallSyst.M_sys
K = OverallSyst.K_sys
C = OverallSyst.C_sys
print('Rank',np.linalg.matrix_rank(M))


f= OverallSyst.f
n = len(f)
dof_select = 362#362 #357
f[dof_select]=1
dof_select = 289 #
f[dof_select+2]=-0.5
# f[dof_select+3]=0.01
f[dof_select+4]=-0.5
# f[dof_select+5]=-0.01
print('Selected DOF: ',OverallSyst.u_names[dof_select])
# K[1,1]=  1.028*10**9
# K[dof_select,dof_select]= 1.028*10**9
# K[dof_select,1]=K[1,dof_select] = -1.028*10**9
import control
A = np.vstack((np.hstack((np.zeros((n,n)),np.eye(n))),
                np.hstack((-np.dot(np.linalg.inv(M),K),-np.dot(np.linalg.inv(M),C)))))
B = np.vstack((np.zeros((n,1)),np.dot(np.linalg.inv(M),f)))
C = np.vstack((np.zeros((int(n/2)-1,12)),np.eye(12),np.zeros((n+int(n/2)-10,12)))).T
D = np.zeros((12,1))
sys = control.ss(A,B,C,D)
# mag, phase, omega = control.freqresp(sys, np.arange(10,2000,0.5))
mag, phase, omega = control.freqresp(sys, np.arange(4,50,0.2)**2*np.pi*2)
#%%
# a=[]
# for i in np.arange(10,3000,10):
#     a.append(control.evalfr(sys, np.complex(imag=i)))
    

w = omega  /2/np.pi

fig1,ax1 = plt.subplots(nrows=1)
for i in range(12):
    H= P2R(mag[i,0,:],phase[i,0,:])
  
    name = OverallSyst.u_names[(int(n/2))-1+i]
    ax1.plot(H.real, H.imag,label='DOF: {}'.format(name))
    ax1.plot(H.real, -H.imag, "r")
ax1.legend()
ax1.set_xlabel('Real')
ax1.set_ylabel('Imag')
for i in range(12):
    H= P2R(mag[i,0,:],phase[i,0,:])
    name = OverallSyst.u_names[(int(n/2))-1+i]
    # fig,ax = plt.subplots(nrows=1)
    # plt.title('Digital filter frequency response for DOF: {}'.format(name))
    # plt.plot(w, 20 * np.log10(abs(H)), 'b')
    # plt.ylabel('Amplitude [dB]', color='b')
    # plt.xlabel('Frequency [rad/sample]')
    # ax2 = ax.twinx()
    # angles = np.unwrap(np.angle(H))
    # plt.plot(w, angles, 'g')
    # plt.ylabel('Angle (radians)', color='g')
    # plt.grid()
    # plt.axis('tight')
    # plt.show()
    fig,ax = plt.subplots(nrows=1)
    plt.title('Digital filter frequency response for DOF: {}'.format(name))
    plt.plot(w, abs(H), 'b')
    plt.ylabel('Amplitude [m/N]', color='b')
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlim([16,2500])
    plt.ylim([10**-10,10**-7])
    plt.xlabel('Frequency [Hz]')
    ax2 = ax.twinx()
    angles = np.unwrap(np.angle(H))
    plt.plot(w, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.show()
    
#%%% Undamped free vibration:
[u,l] = scipy.linalg.eig(K,M)
natural_frequencies = np.sqrt(u)
print(natural_frequencies/np.pi**2/4)
#%%% TIM4 flexibility term
# plt.plot(tim4.point_load_TIM4(np.arange(0,1.01,0.01)))
# #%% Bernoulli Beam deflection
# raise Warning('stop')
# EI_zz=3038*0.01**4*210*10**9#m4*N/m2
# EI_yy=123456789*0.01**4*210*10**9
# W=65000#N / 6500kg
# l=0.6#m
# a=0.5*l # position of load
# b=0.5*l # position of load

# D_eingespannt=W*a**3*b**3/(3*EI_zz*l**3)*1000
# D_einfacherbalken=W*a**2*b**2/(3*EI_zz*l)*1000
# #%% Bernoulli Beam FE
# EI=EI_zz
# k_s =60000/0.0012 #N/m stiffness
# k_is = 0#50000
# M_s =300 # kg sleeper mass B70
# C_s =0.01 #Ns/m damping of sleeper and rail, assumed very low
# I_s = M_s/12*(0.3**2+0.21**2) #kgm^2 assumed sleeper inertia
# C_is = 0.1 # Ns
# n_s = 30 # number of sleepers in model
# M = sparse.spdiags((np.vstack([np.ones((n_s,1))*M_s, np.ones((n_s,1))*I_s])).transpose(),diags=[ 0],m=2*n_s,n=2*n_s)
# C = sparse.spdiags((np.vstack([np.ones((n_s,1))*C_s, np.ones((n_s,1))*C_is])).transpose(),diags=[ 0],m=2*n_s,n=2*n_s)

# K11 = sparse.csc_matrix(sparse.spdiags((np.ones((n_s,1))*np.array([1,2,1])).transpose(),diags=[-1, 0,1],m=n_s,n=n_s)*12*EI/l**3) #displacements
# K11[0,0]=K11[0,0]/2
# K11[int(n_s)-1,int(n_s)-1]=K11[int(n_s)-1,int(n_s)-1]/2
# K11_k = K11 + sparse.eye(n_s)*k_s*10#*100000000
# K11_k[0,0] = K11_k[0,0] + 10**10
# K11_k[n_s-1,n_s-1] =K11_k[n_s-1,n_s-1] + 10**10

# K21 = sparse.spdiags((np.ones((n_s,1))*np.array([-1,1])).transpose(),diags=[-1,1],m=n_s,n=n_s)*6*EI/l**2
# K12 = K21.transpose() 

# K22 = sparse.csc_matrix(sparse.spdiags((np.ones((n_s,1))*np.array([-1,4,-1])).transpose(),diags=[-1, 0,1],m=n_s,n=n_s)*2*EI/l) # rotations
# K22[0,0]=K11[0,0]/2
# K22[int(n_s)-1,int(n_s)-1]=K22[int(n_s)-1,int(n_s)-1]/2
# K22 = K22 + sparse.eye(n_s)*k_is
# K22[0,0] = K22[0,0] + 10**10
# K22[n_s-1,n_s-1] =K22[n_s-1,n_s-1] + 10**10

# K=sparse.bmat([[K11_k, K12],[K21, K22]])
# K=sparse.csc_matrix(K)

# F= np.zeros((2*n_s,1))
# F[int(n_s/2),0]       = -W/2#N
# F[int(n_s/2)+1,0]     = -W/2#N
# F[int(n_s+n_s/2),0]   =  W*l/8#N
# F[int(n_s+n_s/2)+1,0] = -W*l/8#N
# # F[10,0]=100000
# a=linalg.spsolve(K,F)
# plt.plot(a)
# # bb=(a[4:]+a[3:-1]+a[2:-2]+a[1:-3]+a[:-4])/5
# # bb=(bb[4:]+bb[3:-1]+bb[2:-2]+bb[1:-3]+bb[:-4])/5
# # plt.plot(np.arange(4,len(bb)+4),bb)
# b=K.toarray()

# cc=(a[2:int(n_s-1)]-a[0:int(n_s-3)])/2 + a[1:int(n_s-2)]
# plt.plot(np.cumsum(cc))
