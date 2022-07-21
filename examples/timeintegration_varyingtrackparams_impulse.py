# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:21:46 2020

Simplified Vehicle-Track interaction model
@author: e512481
"""
import matplotlib.pyplot as plt
import numpy as np
import sys
# module imports
# from railFE.VehicleModelAssembly import VehicleAssembly
from railFE.TrackModelAssembly import UIC60properties,Pad,Ballast
from railFE.TimoshenkoBeamModel import Timoshenko4,Timoshenko4eb
from railFE.SystemAssembly import OverallSystem
# from railFE.UnitConversions import P2R, R2P
from railFE.Newmark_NL import newmark_nl,TimeIntegrationNM
        
#%% Response for varying track parameters NM
OverallSyst = OverallSystem(support_type = 'pt',n_sleepers=41) # Overall system, default parameters, point support
K_c = OverallSyst.Track.Timoshenko4.railproperties.K_c0 # initial herzian contact stiffness
vary_ballast_params_nm =True
if vary_ballast_params_nm:
    t_integration = {}
    speed = 55
    ti=0
    # current_node
    for k_ballast in np.array([80,100,150,200])*10**6:
        for c_ballast in np.array([50,100,150])*10**3:
            kwargs = {'Ballast':Ballast(K=k_ballast,C=c_ballast)}
            OverallSyst = OverallSystem(support_type= 'eb',**kwargs,n_sleepers=31)
            xi,segment = OverallSyst.timeToLocalReference(ti,speed)
            (K_loc_mod_static,_) = OverallSyst.Local.assembleLocalMatricesStatic(xi,segment,K_c,w_irr=0)
            Y_0 = np.linalg.solve(OverallSyst.K_sys-K_loc_mod_static,OverallSyst.f_sys)
            key = '{}_{}'.format(str(int(k_ballast/10**6)) , str(int(c_ballast/10**3)))
            t_end = (0.6*35/speed)
            time_integration = TimeIntegrationNM(OverallSyst, Y_0, speed,t_end)
            time_integration.newmark_slv()
            
            t_integration[key]=  time_integration
    #%%% Plot the Timeseries for the ballast parameter
    from matplotlib import cm
    
    fig,ax = plt.subplots(figsize = (8,4),ncols =1) 
    for k_key in t_integration.keys():
        time_integration = t_integration[k_key]
        clr = (int(k_key.split('_')[1]))/150*0.7+0.2
        if k_key.split('_')[0]=='80':
           color = cm.Reds(clr)
        elif  k_key.split('_')[0]=='100':
           color = cm.Greens(clr)
        elif  k_key.split('_')[0]=='150':
           color = cm.Greys(clr)
        elif  k_key.split('_')[0]=='200':
           color = cm.Blues(clr)
       
        ax.plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[2,:][:-1], label ='k={} MN/m, c={} kNs/m'.format(*k_key.split('_')),c=color)
        ax.set_ylabel('axle box acceleration[$m/s^2$]')
        ax.scatter([0.3+i*0.6 for i in range(int(time_integration.speed*time_integration.t_end/0.6))],  
                   np.mean(time_integration.A[106])*np.ones(int(time_integration.speed*time_integration.t_end/0.6))*0.25,
                                   marker = 's', color=color,s=40)
    ax.set_xlabel('position [m]')
    # ax.set_ylabel('axle box position [mm]')
    ax.set_xlim([12,20.0])
    fig.legend(loc=7,title='Ballast Parameters') 
    fig.tight_layout()  
    fig.subplots_adjust(right=0.65,wspace=0.25)   
    fig.savefig('../figs/timeintegration_varyingtrackparams_impulse/NM_comparison_point_distributed_support_ballast_variation.png',dpi=300)
 
#%% Response for varying track parameters and speed NM
t_integration = {}
ti=0
for k_ballast in np.array([80,100,150,200])*10**6:
    for speed in np.array([10,15,30,35,40,45,50,55]):
        kwargs = {'Ballast':Ballast(K=k_ballast)}
        OverallSyst = OverallSystem(support_type= 'pt',**kwargs,n_sleepers=51)
        xi,segment = OverallSyst.timeToLocalReference(ti,speed)
        (K_loc_mod_static,_) = OverallSyst.Local.assembleLocalMatricesStatic(xi,segment,K_c,w_irr=0)
        Y_0 = np.linalg.solve(OverallSyst.K_sys-K_loc_mod_static,OverallSyst.f_sys)
        key = '{}_{}'.format(str(int(k_ballast/10**6)) , str(int(speed*3.6)))
        t_end = (0.6*30/speed)
        time_integration = TimeIntegrationNM(OverallSyst, Y_0, speed,t_end)
        time_integration.newmark_slv()
        
        t_integration[key]=  time_integration
#%%% Plot the Timeseries for the ballast parameter and speed
from matplotlib import cm

speedsdict = [str(i) for i in sorted(set([int(i.split('_')[1]) for i in t_integration.keys()]))]
speedsdict = dict([(i,j) for (j,i) in enumerate(speedsdict)])

fig,ax = plt.subplots(figsize = (12,8),nrows = len(speedsdict),ncols =1,sharex=True,sharey=True,dpi=200) 
for k_key in t_integration.keys():
    time_integration = t_integration[k_key]
    clr = 0.8#(int(k_key.split('_')[1]))/200*0.7+0.2
    if k_key.split('_')[0]=='80':
       color = cm.Reds(clr)
    elif  k_key.split('_')[0]=='100':
       color = cm.Greens(clr)
    elif  k_key.split('_')[0]=='150':
       color = cm.Purples(clr)
    elif  k_key.split('_')[0]=='200':
       color = cm.Blues(clr)
    delta = len(np.arange(0,time_integration.t_end,time_integration.dt))-len(time_integration.A[2,:])
    ax[speedsdict[k_key.split('_')[1]]].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed-0.25*time_integration.speed,
            time_integration.A[2,:][:delta], label ='k={} MN/m, v={} km/h'.format(*k_key.split('_')),c=color)
    ax[speedsdict[k_key.split('_')[1]]].set_ylabel('ABA [$m/s^2$]')
    # ax.scatter([0.3+i*0.6 for i in range(int(time_integration.speed*time_integration.t_end/0.6))],  
    #            np.mean(time_integration.A[106])*np.ones(int(time_integration.speed*time_integration.t_end/0.6))*0.25,
    #                            marker = 's', color=color,s=40)
    ax[speedsdict[k_key.split('_')[1]]].set_title('time series for {} m/s'.format(k_key.split('_')[1]))
ax[-1].set_xlabel('position [m]')
# ax.set_ylabel('axle box position [mm]')
ax[-1].set_xlim([0,3])
fig.legend(loc=7,title='Ballast Parameters') 
fig.tight_layout()  
fig.subplots_adjust(right=0.65,wspace=0.25)   
fig.savefig('../figs/timeintegration_varyingtrackparams_impulse/NM_comparison_point_distributed_support_ballast_variation_speed.png',dpi=300)

sys.exit()
#%% Additional Plots
import pandas as pd
import seaborn as sns
speed=[]
amplitude = []
stiffness = []
for k_key in t_integration.keys():
    time_integration = t_integration[k_key]
    amplitude.append(time_integration.A[2,:][int(len(time_integration.A)*0.5):].max()-time_integration.A[2,:][int(len(time_integration.A)*0.5):].min())
    speed.append(int(k_key.split('_')[1]))
    stiffness.append(int(k_key.split('_')[0]))
    
surface = pd.DataFrame(np.array([speed,amplitude,stiffness]).T,columns=['speed [$km/h$]','amplitude [$m/s^2$]','Stiffness [$MN/m^2$]']) 
plt.figure()
sns.lineplot(x='speed [$km/h$]',y='amplitude [$m/s^2$]',hue='Stiffness [$MN/m^2$]',data=surface)

#%%% Run integration for different speeds:
t_integration = {}
for speed in [10,20,30,40,50,60]:
    OverallSyst = OverallSystem(support_type = 'eb',n_sleepers=31)
    
    K_c = OverallSyst.Track.Timoshenko4.railproperties.K_c0
    # current_node
    ti=0
    xi,segment = OverallSyst.timeToLocalReference(ti,speed)
    (K_loc_mod_static,_) = OverallSyst.Local.assembleLocalMatricesStatic(xi,segment,K_c,w_irr=0)
    Y_0 = np.linalg.solve(OverallSyst.K_sys-K_loc_mod_static,OverallSyst.f_sys)
    
    time_integration = TimeIntegrationNM(OverallSyst, Y_0, speed)
    newmark_nl(time_integration) 
    t_integration[str(speed)] = time_integration
#%%%% Plotting
fig,ax = plt.subplots(nrows = 1,sharex = True)
for speed in t_integration.keys():
    time_integration= t_integration[speed]
    delta = len(np.arange(0,time_integration.t_end,time_integration.dt))-len(time_integration.U[1,:])
    ax.plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[1,:][:delta]*1000,label = '{} m/s'.format(speed))
ax.scatter([0.3+i*0.6 for i in range(int(time_integration.speed*time_integration.t_end/0.6))], 
                                np.mean(time_integration.U[106])*np.ones(int(time_integration.speed*time_integration.t_end/0.6))*1000*1.4,
                               marker = 's', color='k',s=40)
ax.set_xlabel('position [m]')
ax.set_ylabel('axle box position [mm]')
ax.set_xlim([8,12])
ax.legend()
fig.tight_layout()
fig.savefig('../figs/timeintegration_varyingtrackparams_impulse/timeintegration.png',dpi=300)
