# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:09:12 2020

@author: e512481
"""
import numpy as np
import matplotlib.pyplot as plt
from _SimplifiedVTIM_edited import *
#%%% Response for varying track parameters:
output_fr = {}
for k_ballast in np.array([40,100,200])*10**6:
    for c_ballast in np.array([10,50,100,150])*10**3:
        kwargs = {'Ballast':Ballast(K=k_ballast,C=c_ballast)}
        OverallSyst,M,K,C,f, dof_rail_mid_span,dof_rail_support,dof_sleeper,dofs_rail,dofs_sleeper = system_matrix(support_type= 'eb',**kwargs)
        sys_eb =  assemble_state_space(K,M,C,f,[dof_rail_mid_span , dof_rail_support,dof_sleeper])
        mag, phase, omega = control.freqresp(sys_eb, np.arange(4,50,0.2)**2*np.pi*2)
        key = '{}_{}'.format(str(int(k_ballast/10**6)) , str(int(c_ballast/10**3)))
        output_fr[key]= (sys_eb ,mag, phase, omega)
#%%%% Plot the Peak response for the ballast parameter
from matplotlib import cm

fig,ax = plt.subplots(figsize = (8,4),ncols =2) 
for k_key in output_fr.keys():
    (sys_eb ,mag, phase, omega) = output_fr[k_key]
    clr = (int(k_key.split('_')[1]))/150*0.7+0.2
    if k_key.split('_')[0]=='40':
       color = cm.Reds(clr)
    elif  k_key.split('_')[0]=='80':
       color = cm.Greens(clr)
    elif  k_key.split('_')[0]=='120':
       color = cm.Greys(clr)
    elif  k_key.split('_')[0]=='200':
       color = cm.Blues(clr)
    # Get both three chosen responses
    H_midspan= P2R(mag[0,0,:],phase[0,0,:])
    H_support= P2R(mag[1,1,:],phase[1,1,:])
    ax[0].plot(w, abs(H_midspan), label ='k={} MN/m, c={} kNs/m'.format(*k_key.split('_')),c=color)
    ax[1].plot(w, abs(H_support),c=color)#, label ='EB, Rail support {}MN/m, {}kNs/m'.format(*k_key.split('_')),c=color)
for axi in ax:  
    axi.set_xscale('log')
    axi.set_yscale('log')
    axi.set_xlim([16,2500])
    # axi.set_ylim([10**-10,2*10**-7])
    axi.set_xlabel('Frequency [Hz]')
    axi.grid()
    axi.grid(which = 'minor', linestyle ='--',linewidth=0.3)
    
ax[0].set_title('FR for excitation mid span')
ax[1].set_title('FR for excitation over sleeper')
ax[0].set_ylabel('Amplitude [m/N]', color='b')
ax[0].set_ylim([2*10**-10,2*10**-8])
ax[1].set_ylim([2*10**-10,2*10**-8])

fig.legend(loc=7,title='Ballast Parameters') 
fig.tight_layout()  
fig.subplots_adjust(right=0.65,wspace=0.25)   
#%%% Response for varying pad parameters:
output_fr2 = {}
for k_pad in np.array([150,350,700])*10**6:
    for c_pad in np.array([20,50,100])*10**3:
        kwargs = {'Pad':Pad(K=k_pad,C=c_pad)}
        OverallSyst,M,K,C,f, dof_rail_mid_span,dof_rail_support,dof_sleeper,dofs_rail,dofs_sleeper = system_matrix(support_type= 'eb',**kwargs)
        sys_eb =  assemble_state_space(K,M,C,f,[dof_rail_mid_span , dof_rail_support,dof_sleeper])
        mag, phase, omega = control.freqresp(sys_eb, np.arange(4,50,0.2)**2*np.pi*2)
        key = '{}_{}'.format(str(int(k_pad/10**6)) , str(int(c_pad/10**3)))
        output_fr2[key]= (sys_eb ,mag, phase, omega)
#%%%% Plot the Peak response for the pad parameter
from matplotlib import cm

fig,ax = plt.subplots(figsize = (8,4),ncols =2) 
for k_key in output_fr2.keys():
    (sys_eb ,mag, phase, omega) = output_fr2[k_key]
    clr = (int(k_key.split('_')[1]))/100*0.7+0.2
    if k_key.split('_')[0]=='150':
       color = cm.Reds(clr)
    elif  k_key.split('_')[0]=='350':
       color = cm.Greens(clr)
    elif  k_key.split('_')[0]=='700':
       color = cm.Blues(clr)
    # Get both three chosen responses
    H_midspan= P2R(mag[0,0,:],phase[0,0,:])
    H_support= P2R(mag[1,1,:],phase[1,1,:])
    ax[0].plot(w, abs(H_midspan), label ='k={} MN/m, c={} kNs/m'.format(*k_key.split('_')),c=color)
    ax[1].plot(w, abs(H_support),c=color)#, label ='EB, Rail support {}MN/m, {}kNs/m'.format(*k_key.split('_')),c=color)
for axi in ax:  
    axi.set_xscale('log')
    axi.set_yscale('log')
    axi.set_xlim([16,2500])
    # axi.set_ylim([10**-10,2*10**-7])
    axi.set_xlabel('Frequency [Hz]')
    axi.grid()
    axi.grid(which = 'minor', linestyle ='--',linewidth=0.3)
    
ax[0].set_title('FR for excitation mid span')
ax[1].set_title('FR for excitation over sleeper')
ax[0].set_ylabel('Amplitude [m/N]', color='b')
ax[0].set_ylim([2*10**-10,2*10**-8])
ax[1].set_ylim([2*10**-10,2*10**-8])

# ax[0].legend()
fig.legend(loc=7,title='Pad Parameters') 
fig.tight_layout()  
fig.subplots_adjust(right=0.65,wspace=0.25) 