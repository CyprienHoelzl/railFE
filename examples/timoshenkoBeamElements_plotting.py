# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:21:46 2020

Timoshenko Beam elements. Plotting and comparing shape functions.

@author: Cyprien Hoelzl
"""
import matplotlib.pyplot as plt
import numpy as np
from railFE.TrackModelAssembly import UIC60properties,Pad
from railFE.TimoshenkoBeamModel import Timoshenko4,Timoshenko4eb

#%% Plotting       
def plot_shapefunctions():
    """
    Plot the interpolation shape function for TIM4

    Returns
    -------
    fig : matplotlib figure
    ax : axes

    """
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
    fig.tight_layout()
    return fig,ax

def plot_shapefunctions_el():
    """
    Plot the interpolation shape function for TIM4eb (timoshenko on elastic bedding)

    Returns
    -------
    fig : matplotlib figure
    ax : axes

    """
    UIC60props = UIC60properties()
    tim4el = Timoshenko4eb(UIC60props,Pad(),0.14)
    
    fig,ax = plt.subplots(figsize=(4,3))
    ax.plot(tim4el.N_w1(np.arange(0,1.1,0.1)),label = r'$N_{w_1}(\xi)$')
    ax.plot(tim4el.N_t1(np.arange(0,1.1,0.1)),label = r'$N_{\theta_1}(\xi)$')
    ax.plot(tim4el.N_w2(np.arange(0,1.1,0.1)),label = r'$N_{w_2}(\xi)$')
    ax.plot(tim4el.N_t2(np.arange(0,1.1,0.1)),label = r'$N_{\theta_2}(\xi)$')
    ax.plot(tim4el.N_s(np.arange(0,1.1,0.1)),'--',label = r'$N_{s}(\xi)$')
    ax.legend()
    ax.set_xlabel(r"$\xi = x/L$")
    ax.set_ylabel('$[-]$')
    ax.set_title('Interpolation Functions for TIM4 elastic bedding')
    fig.tight_layout()
    return fig,ax
def plot_comparison():
    """
    Plot the interpolation function comparison of TIM4 and TIM4eb

    Returns
    -------
    fig : matplotlib figure
    ax : axes

    """
    fig,ax = plt.subplots(figsize=(7,3),ncols=2)
    fig.suptitle('TIM4 difference for distributed stiffness')
    for i in np.arange(200,1200,200):
        UIC60props = UIC60properties()
        tim4 = Timoshenko4(UIC60props,0.5)
        tim4el = Timoshenko4eb(UIC60props,Pad(K = i*10**6),0.5)
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
    """
    Plot the interpolation function for the sleeper displacement for the flexible track element (Timoshenko4eb)

    Returns
    -------
    fig : figure with 3d surface plot.
    ax : axes

    """
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
        tim4el = Timoshenko4eb(UIC60props,Pad(K=i*10**6),0.5)
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
    ax.set_title('Interpolation function for the sleeper displacement \nfor flexible track element')
    fig.tight_layout()
    return fig,ax  
#%% Application
if __name__ == "__main__":
    fig,ax = plot_shapefunctions()
    fig.savefig('../figs/timoshenkoBeamElements/shapeFunctions_Timoshenko4.png',dpi=200)
    fig,ax = plot_shapefunctions_el()
    fig.savefig('../figs/timoshenkoBeamElements/shapeFunctions_Timoshenko4_elasticbedding.png',dpi=200)
    fig,ax = plot_comparison()
    fig.savefig('../figs/timoshenkoBeamElements/shapeFunctions_comparison_Timoshenko4_elasticbedding.png',dpi=200)
    fig,ax = plot_sleeperstiffnessNs()
    fig.savefig('../figs/timoshenkoBeamElements/shapeFunctions_sleeper_stiffness_sleeperdisplacement.png',dpi=200)