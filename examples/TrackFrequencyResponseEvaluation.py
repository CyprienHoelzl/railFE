# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:09:12 2020

Frequency response for varying load positions and track parameters 

@author: Cyprien Hoelzl
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as markers
import control
from scipy.signal import find_peaks

from railFE.TrackModelAssembly import Ballast, Pad  
from railFE.SystemAssembly import system_matrix,assemble_state_space
from railFE.UnitConversions import P2R

omega = np.arange(4,50,0.2)**2*np.pi*2  # frequency at which the system response is evaluated
w = omega  /2/np.pi # frequency in Hz

#%% Frequency Response at sleeper mid span and over sleeper for excitation at mid span and over sleeper
OverallSyst,M,K,C,f, dof_rail_mid_span,dof_rail_support,dof_sleeper,dofs_rail,dofs_sleeper = system_matrix(support_type= 'eb')
sys_eb =  assemble_state_space(K,M,C,f,[dof_rail_mid_span , dof_rail_support,dof_sleeper])#,dofs_rail ])
OverallSyst,M,K,C,f, dof_rail_mid_span,dof_rail_support,dof_sleeper,dofs_rail,dofs_sleeper = system_matrix(support_type= 'pt')
sys_pt =  assemble_state_space(K,M,C,f,[dof_rail_mid_span , dof_rail_support,dof_sleeper])

mag, phase, omega = control.freqresp(sys_eb,omega)
mag_pt, phase_pt, omega_pt = control.freqresp(sys_pt, omega)
#%%% Plot for vertical response in frequency domain for rail and sleeper
fig,ax = plt.subplots(figsize = (8,3.8),ncols =mag.shape[1])
# For each excitation
for i in range(mag.shape[1]):
    # Get both three chosen responses
    H_midspan= P2R(mag[0,i,:],phase[0,i,:])
    H_support= P2R(mag[1,i,:],phase[1,i,:])
    H_sleeper= P2R(mag[2,i,:],phase[2,i,:])
    H_midspan_pt= P2R(mag_pt[0,i,:],phase_pt[0,i,:])
    H_support_pt= P2R(mag_pt[1,i,:],phase_pt[1,i,:])
    H_sleeper_pt= P2R(mag_pt[2,i,:],phase_pt[2,i,:])
    ax[i].plot(w, abs(H_midspan_pt), label ='PT (point sleeper support), Rail midspan')
    ax[i].plot(w, abs(H_midspan), label ='EB (distributed sleeper support), Rail midspan')
    ax[i].plot(w, abs(H_support_pt), label ='PT, Rail support')
    ax[i].plot(w, abs(H_support), label ='EB, Rail support')
    ax[i].plot(w, abs(H_sleeper_pt), label ='PT, Sleeper')
    ax[i].plot(w, abs(H_sleeper), label ='EB, Sleeper')
    
    ax[i].set_xscale('log')
    ax[i].set_yscale('log')
    ax[i].set_xlim([16,2500])
    ax[i].set_ylim([10**-10,2*10**-8])
    ax[i].set_xlabel('Frequency [Hz]')
    ax[i].grid()
    ax[i].grid(which = 'minor', linestyle ='--',linewidth=0.3)
    
ax[0].set_title('FR for excitation mid span')
ax[1].set_title('FR for excitation over sleeper')
ax[0].set_ylabel('Amplitude [m/N]', color='b')

ax[0].legend()
fig.tight_layout()
fig.savefig('../figs/trackFrequencyResponseEvaluation/FrequencyResponse_point_distributed_support.png',dpi=300)

#%% Calculate Frequency response for varying track parameters:
output_fr = {}
for k_ballast in np.array([40,100,200])*10**6: # ballast stiffness
    for c_ballast in np.array([10,50,100,150])*10**3: # ballast damping
        kwargs = {'Ballast':Ballast(K=k_ballast,C=c_ballast)}
        OverallSyst,M,K,C,f, dof_rail_mid_span,dof_rail_support,dof_sleeper,dofs_rail,dofs_sleeper = system_matrix(support_type= 'eb',**kwargs)
        sys_eb =  assemble_state_space(K,M,C,f,[dof_rail_mid_span , dof_rail_support,dof_sleeper]) # state space formulation of the system matrices
        mag, phase, omega = control.freqresp(sys_eb, omega) # frequency response calculation
        key = '{}_{}'.format(str(int(k_ballast/10**6)) , str(int(c_ballast/10**3)))
        output_fr[key]= (sys_eb ,mag, phase, omega) # saving results to dictionary with key '{stiffness}_{damping}'
#%%% Plot the Peak response for the varying ballast parameter
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
ax[0].set_ylabel('Amplitude [m/N]', color='k')
ax[0].set_ylim([2*10**-10,2*10**-8])
ax[1].set_ylim([2*10**-10,2*10**-8])

fig.legend(loc=7,title='Ballast Parameters') 
fig.tight_layout()  
fig.subplots_adjust(right=0.65,wspace=0.25)   
fig.savefig('../figs/trackFrequencyResponseEvaluation/FrequencyResponse_ChangingBallastParams.png', dpi =300)
#%% Response for varying pad parameters:
output_fr2 = {}
for k_pad in np.array([150,350,700])*10**6:
    for c_pad in np.array([20,50,100])*10**3:
        kwargs = {'Pad':Pad(K=k_pad,C=c_pad)}
        OverallSyst,M,K,C,f, dof_rail_mid_span,dof_rail_support,dof_sleeper,dofs_rail,dofs_sleeper = system_matrix(support_type= 'eb',**kwargs)
        sys_eb =  assemble_state_space(K,M,C,f,[dof_rail_mid_span , dof_rail_support,dof_sleeper]) # state space formulation of the system matrices
        mag, phase, omega = control.freqresp(sys_eb, omega) # frequency response calculation
        key = '{}_{}'.format(str(int(k_pad/10**6)) , str(int(c_pad/10**3)))
        output_fr2[key]= (sys_eb ,mag, phase, omega) # saving results to dictionary with key '{stiffness}_{damping}'
#%%% Plot the Peak response for the varying pad parameter
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
fig.savefig('../figs/trackFrequencyResponseEvaluation/FrequencyResponse_ChangingPadParams.png', dpi =300)



#%% Modal responses
OverallSyst,M,K,C,f, dof_rail_mid_span,dof_rail_support,dof_sleeper,dofs_rail,dofs_sleeper = system_matrix(support_type= 'eb')
dofs_rail_rot = np.arange(len(OverallSyst.u_names))[[True if ('t_rail_' in i) else False for i in OverallSyst.u_names]]

n_sleepers_modes = 16 #number of sleeper modes considered for plotting
selected_range = range(int((OverallSyst.n_sleepers-1-n_sleepers_modes )/2),int((OverallSyst.n_sleepers-1+n_sleepers_modes )/2))
selected_raildofs = dofs_rail[[True if int(i.split('_')[2]) in selected_range else False for i in np.array(OverallSyst.u_names)[dofs_rail]]]
selected_raildofs_rot = dofs_rail_rot[[True if int(i.split('_')[2]) in selected_range else False for i in np.array(OverallSyst.u_names)[dofs_rail_rot]]]

selected_sleeperdofs = dofs_sleeper[[True if int(i.split('_')[2]) in selected_range else False for i in np.array(OverallSyst.u_names)[dofs_sleeper]]]
# Assemlke State space for loading vector and responses
sys_eb =  assemble_state_space(K,M,C,f,np.concatenate((selected_raildofs,selected_raildofs_rot , selected_sleeperdofs),axis=0).reshape((-1,1)))#,dofs_rail ])

# Fresresp
mag, phase, omega = control.freqresp(sys_eb, omega)

# peak_frequencies = np.array([95,240,530,1100])
#%%% Plot Modal response shapes
max_number_mode_peaks = 8

square_marker=markers.MarkerStyle(marker='s')
for loading in [0,1]: # 0:midspan between sleepers, 1:over sleeper
    peak_frequency_indexes,_ = find_peaks(abs(P2R(mag[0,loading,:],phase[0,loading,:])))
    number_mode_peaks = min(max_number_mode_peaks,len(peak_frequency_indexes))
    fig,ax = plt.subplots(figsize=(6,5),nrows = number_mode_peaks,ncols=2,sharex = True)
    for mode_index in range(number_mode_peaks):
        mode = peak_frequency_indexes[mode_index]
        H_midspan= P2R(mag[:,loading,mode],phase[:,loading,mode])
        no_raildofs = len(selected_raildofs)+len(selected_raildofs_rot)
        abs_rail = abs(H_midspan[:no_raildofs])[:len(selected_raildofs)]
        real_rail = np.real(H_midspan[:no_raildofs])[:len(selected_raildofs)]
        abs_sleepers = abs(H_midspan)[no_raildofs:]
        real_sleepers = np.real(H_midspan)[no_raildofs:]
        for i in range(n_sleepers_modes-1):
            xi = np.arange(0,1.05,0.02)
            x= np.reshape([xi*0.16,0.16+xi*0.22,0.38+xi*0.22],(-1))+0.6*i-0.5*n_sleepers_modes*0.6+0.3-0.08+0.3*loading
            y = np.reshape([OverallSyst.Track.Timoshenko4eb.N_t1(xi)*H_midspan[len(selected_raildofs)+0+i*3]+
                            OverallSyst.Track.Timoshenko4eb.N_t2(xi)*H_midspan[len(selected_raildofs)+1+i*3]+
                            OverallSyst.Track.Timoshenko4eb.N_w1(xi)*H_midspan[0+i*3]+
                            OverallSyst.Track.Timoshenko4eb.N_w2(xi)*H_midspan[1+i*3] ,        
                            OverallSyst.Track.Timoshenko4.N_t1(xi)*H_midspan[len(selected_raildofs)+1+i*3]+
                            OverallSyst.Track.Timoshenko4.N_t2(xi)*H_midspan[len(selected_raildofs)+2+i*3]+
                            OverallSyst.Track.Timoshenko4.N_w1(xi)*H_midspan[1+i*3]+
                            OverallSyst.Track.Timoshenko4.N_w2(xi)*H_midspan[2+i*3] ,        
                            OverallSyst.Track.Timoshenko4.N_t1(xi)*H_midspan[len(selected_raildofs)+2+i*3]+
                            OverallSyst.Track.Timoshenko4.N_t2(xi)*H_midspan[len(selected_raildofs)+3+i*3]+
                            OverallSyst.Track.Timoshenko4.N_w1(xi)*H_midspan[2+i*3]+
                            OverallSyst.Track.Timoshenko4.N_w2(xi)*H_midspan[3+i*3]],(-1))
                
            ax[mode_index,0].plot(x,-abs(y)*10**9,'k')    
            ax[mode_index,1].plot(x,-np.real(y)*10**9,'k')    
        # x = np.reshape([np.array([0,0.16,0.38])+i*0.6-0.5*n_sleepers_modes*0.6+0.3-0.08 for i in        range(n_sleepers_modes)],(-1))
        # ax[mode_index,0].plot(x,-abs_rail*10**9 )
        # ax[mode_index,1].plot(x,-real_rail*10**9 )
        
        # ax[mode_index].plot(sleeper_position)   
        ax[mode_index,0].scatter([i*0.6-0.5*n_sleepers_modes*0.6+0.3+0.3*loading for i in range(n_sleepers_modes)], 
                           -0.08-abs_sleepers*10**9,
                           marker = 's', color='k',s=15)
        ax[mode_index,1].scatter([i*0.6-0.5*n_sleepers_modes*0.6+0.3+0.3*loading for i in range(n_sleepers_modes)], -0.08-real_sleepers*10**9,marker = 's', color='k',s=15)
    
        ax[mode_index,0].set_xlim([-4.64+0.3*loading,4.68+0.3*loading])
        ax[mode_index,1].set_xlim([-4.64+0.3*loading,4.68+0.3*loading])
        
        min_y = min(ax[mode_index,0].get_ylim()[0],ax[mode_index,1].get_ylim()[0])
        max_y = max(ax[mode_index,0].get_ylim()[1],ax[mode_index,1].get_ylim()[1])
        if mode==0:
            ax[mode_index,0].set_ylim(np.array([min_y,max_y])*1.1)
            ax[mode_index,1].set_ylim(np.array([min_y,max_y])*1.1)
        else:
            ax[mode_index,0].set_ylim(np.array([min_y,max_y])*1.1)
            ax[mode_index,1].set_ylim(np.array([min_y,max_y])*1.1)
        ax[mode_index,0].set_ylabel('$[mm/MN]$')
        ax[mode_index,0].annotate('${} [Hz]$'.format(w[mode].round(2)), 
                             xy=(0.05, 0.05), xycoords='axes fraction', fontsize=8,
                    horizontalalignment='left', verticalalignment='bottom')
    
    ax[-1,0].set_xlabel('Position [m]')
    ax[-1,1].set_xlabel('Position [m]')
    ax[0,0].set_title('Amplitude')
    ax[0,1].set_title('Shape')
    plt.tight_layout()
    fig.align_ylabels()
    
    if loading==0:
        fig.savefig('../figs/trackFrequencyResponseEvaluation/mode_shapes_load_midspan.png',dpi=300)
    else:
        fig.savefig('../figs/trackFrequencyResponseEvaluation/mode_shapes_load_oversleeper.png',dpi=300)