# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:21:46 2020

Simplified Vehicle-Track interaction model
@author: e512481
"""
import matplotlib.pyplot as plt
import numpy as np
# module imports
# from railFE.VehicleModelAssembly import VehicleAssembly
from railFE.TrackModelAssembly import UIC60properties,Pad,Ballast
from railFE.TimoshenkoBeamModel import Timoshenko4,Timoshenko4eb
from railFE.SystemAssembly import OverallSystem
# from railFE.UnitConversions import P2R, R2P
from railFE.Newmark_NL import newmark_nl,TimeIntegrationNM
    
def random_track_noise(ti):
    from sklearn.preprocessing import MinMaxScaler
    x= np.linspace(0,1,10000)
    for i in range(20):    
        y = np.convolve(np.random.randint(0,100)*np.sin(np.random.randint(0,100)*x), np.random.randint(0,100)*np.cos(np.random.randint(0,100)*x), 'same')
        y = MinMaxScaler(feature_range=(-5,5)).fit_transform(y.reshape(-1, 1)).reshape(-1)
#%% Perform time integration
OverallSyst = OverallSystem(support_type = 'pt',n_sleepers=41) # Overall system, default parameters, point support
K_c = OverallSyst.Track.Timoshenko4.railproperties.K_c0 # initial herzian contact stiffness
speed = 18 # m/s vehicle speed
ti=0 
t_end = (0.6*15/speed)

xi,segment = OverallSyst.timeToLocalReference(ti,speed) # Current position
(K_loc_mod_static,_) = OverallSyst.Local.assembleLocalMatricesStatic(xi,segment,K_c,w_irr=0) # Current initial K_loc
Y_0 = np.linalg.solve(OverallSyst.K_sys-K_loc_mod_static,OverallSyst.f_sys) # Initial Y_0

time_integration = TimeIntegrationNM(OverallSyst, Y_0, speed,t_end)
time_integration.newmark_slv() 
#%% Plot integration result ABA
#%%% UVA for Axle
fig,ax = plt.subplots(nrows = 3,sharex = True)
delta = len(np.arange(0,time_integration.t_end,time_integration.dt))-len(time_integration.U[1,:])
ax[0].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[2,:][:delta])
# ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,np.diff(time_integration.U[1,:-1]/time_integration.dt))
ax[0].set_ylabel('axle box \ndisplacement \n[$m$]')
ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.V[2,:][:delta])
ax[1].set_ylabel('axle box \nvelocity \n[$m/s$]')
ax[2].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[2,:][:delta])
ax[2].set_xlabel('position [$m$]')
ax[2].set_ylabel('axle box \nacceleration \n[$m/s^2$]')

ax[0].set_ylim(np.quantile(time_integration.U[2,:][:delta],0.000001),np.quantile(time_integration.U[2,:][1:delta],0.999))
ax[1].set_ylim(np.quantile(time_integration.V[2,:][:delta],0.001),np.quantile(time_integration.V[2,:][1:delta],0.999))
ax[2].set_ylim(np.quantile(time_integration.A[2,:][:delta],0.001),np.quantile(time_integration.A[2,:][1:delta],0.999))
ax[0].scatter([0.3+i*0.6 for i in range(int(time_integration.speed*time_integration.t_end/0.6))], 
                               0.00009*np.ones(int(time_integration.speed*time_integration.t_end/0.6)),
                               marker = 's', color='k',s=15)
ax[2].scatter([0.3+i*0.6 for i in range(int(time_integration.speed*time_integration.t_end/0.6))], 
                               0*np.ones(int(time_integration.speed*time_integration.t_end/0.6)),
                               marker = 's', color='k',s=15)
fig.align_ylabels()
fig.align_ylabels()
fig.tight_layout()
fig.savefig('../figs/timeintegration_default_track_impulse/time_integration_timeseries_axle.png',dpi=300)

fig.axes[0].set_xlim(4,6)
fig.savefig('../figs/timeintegration_default_track_impulse/time_integration_timeseries_axle_Zoomed.png',dpi=300)

#%%% UVA for Bogie
fig,ax = plt.subplots(nrows = 3,sharex = True)
ax[0].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[1,:][:delta]*1000)
ax[0].set_ylabel('Bogie \ndisplacement \n[$mm$]')
ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.V[1,:][:delta])
ax[1].set_ylabel('Bogie \nvelocity \n[$m/s$]')
ax[2].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[1,:][:delta])
ax[2].set_xlabel('position [$m$]')
ax[2].set_ylabel('Bogie \nacceleration [$m/s^2$]')
# ax[0].scatter([0.3+i*0.6 for i in range(int(time_integration.speed*time_integration.t_end/0.6))], 
#                                -2.45*np.ones(int(time_integration.speed*time_integration.t_end/0.6)),
#                                marker = 's', color='k',s=15)
# ax[1].scatter([0.3+i*0.6 for i in range(int(time_integration.speed*time_integration.t_end/0.6))], 
#                                -2.45*np.ones(int(time_integration.speed*time_integration.t_end/0.6)),
#                                marker = 's', color='k',s=15)
ax[2].scatter([0.3+i*0.6 for i in range(int(time_integration.speed*time_integration.t_end/0.6))], 
                               -0.000*np.ones(int(time_integration.speed*time_integration.t_end/0.6)),
                               marker = 's', color='k',s=15)
fig.align_ylabels()
fig.tight_layout()
fig.savefig('../figs/timeintegration_default_track_impulse/time_integration_timeseries_bogie.png',dpi=300)

#%%% UVA for Body
fig,ax = plt.subplots(nrows = 3,sharex = True)
ax[0].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[0,:][:delta]*1000)
ax[0].set_ylabel('Body \ndisplacement \n[$mm$]')
ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.V[0,:][:delta])
ax[1].set_ylabel('Body \nvelocity \n[$m/s$]')
ax[2].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[0,:][:delta])
ax[2].set_xlabel('position [$m$]')
ax[2].set_ylabel('Body \nacceleration [$m/s^2$]')
# ax[0].scatter([0.3+i*0.6 for i in range(int(time_integration.speed*time_integration.t_end/0.6))], 
#                                -2.45*np.ones(int(time_integration.speed*time_integration.t_end/0.6)),
#                                marker = 's', color='k',s=15)
ax[2].scatter([0.3+i*0.6 for i in range(int(time_integration.speed*time_integration.t_end/0.6))], 
                               -0.000*np.ones(int(time_integration.speed*time_integration.t_end/0.6)),
                               marker = 's', color='k',s=15)
fig.align_ylabels()
fig.tight_layout()
fig.savefig('../figs/timeintegration_default_track_impulse/time_integration_timeseries_body.png',dpi=300)

       
#%%% Plot w DOFs on rails and sleeper
fig,ax = plt.subplots(nrows = 3,figsize= (8,5),sharex = True)
# ax[0].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[0,:][:-1])
ax[0].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[100,:][:-2], label = OverallSyst.u_names[100])
ax[0].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[102,:][:-2], label = OverallSyst.u_names[102])
ax[0].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[104,:][:-2], label = OverallSyst.u_names[104])
ax[0].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[107,:][:-2], label = OverallSyst.u_names[107])
ax[0].set_ylabel('displacement [m]')
# ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.V[0,:][:-1])
ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.V[100,:][:-2])
ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.V[102,:][:-2])
ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.V[104,:][:-2])
ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.V[107,:][:-2])
ax[1].set_ylabel('velocity [m/s]')
# ax[2].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[0,:][:-1])
ax[2].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[100,:][:-2])
ax[2].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[102,:][:-2])
ax[2].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[104,:][:-2])
ax[2].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[107,:][:-2])
ax[2].set_xlabel('position [m]')
ax[2].set_ylabel('acceleration [m/s^2]')
ax[0].legend(ncol=2)
ax[0].scatter([0.3+i*0.6 for i in range(int(time_integration.speed*time_integration.t_end/0.6))], 
                               -0.0000145*np.ones(int(time_integration.speed*time_integration.t_end/0.6)),
                               marker = 's', color='k',s=15)
ax[0].set_xlim(4,6)
fig.align_ylabels()
fig.tight_layout()
fig.savefig('../figs/timeintegration_default_track_impulse/time_integration_timeseries_railSleeper.png',dpi=300)

#%%% Plot t DOFs on rail and sleeper
fig,ax = plt.subplots(nrows = 3,figsize= (8,5),sharex = True)
# ax[0].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[0,:][:-1])
ax[0].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[133,:][:-2], label = OverallSyst.u_names[133])
ax[0].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[135,:][:-2], label = OverallSyst.u_names[135])
ax[0].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[137,:][:-2], label = OverallSyst.u_names[137])
ax[0].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[140,:][:-2], label = OverallSyst.u_names[140])
ax[0].set_ylabel('position [m]')
# ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.V[0,:][:-1])
ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.V[133,:][:-2])
ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.V[135,:][:-2])
ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.V[137,:][:-2])
ax[1].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.V[140,:][:-2])
ax[1].set_ylabel('velocity [m/s]')
# ax[2].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[0,:][:-1])
ax[2].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[133,:][:-2])
ax[2].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[135,:][:-2])
ax[2].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[137,:][:-2])
ax[2].plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.A[140,:][:-2])
ax[2].set_xlabel('position [m]')
ax[2].set_ylabel('acceleration [m/s^2]')

ax[0].scatter([0.3+i*0.6 for i in range(int(time_integration.speed*time_integration.t_end/0.6))], 
                               -0.0000145*np.ones(int(time_integration.speed*time_integration.t_end/0.6)),
                               marker = 's', color='k',s=15)
ax[0].legend(ncol=2)
ax[0].set_xlim(4,6)
fig.align_ylabels()
fig.tight_layout()
fig.savefig('../figs/timeintegration_default_track_impulse/time_integration_timeseries_rail.png',dpi=300)

# =============================================================================
# #%% Plot rail vibration modes
# plt.figure()
# for i in range(10):
#     plt.plot(np.arange(0,time_integration.t_end,time_integration.dt)*time_integration.speed,time_integration.U[OverallSyst.repeated_segments['12']['modal_indexes'][i],:delta].T,label='{}th mode'.format(i+1))
# plt.legend()
# plt.xlabel('position [$m$]')
# plt.xlim(0,2)
# =============================================================================

# =============================================================================
# #%% Testing time integration / Plotting
# # u = np.linalg.solve(OverallSyst.K_sys_upd,OverallSyst.f_sys)
# # plt.figure()
# # x=np.append(np.hstack([np.array([0,0.22,0.38])+i*0.6 for i in range(OverallSyst.n_sleepers)]),18.6)
# # # x= np.reshape([xi*0.16,0.16+xi*0.22,0.38+xi*0.22],(-1))+0.6*0-0.5*n_sleepers_modes*0.6+0.3-0.08+0.3*loading
# # plt.plot(x,u[[i for i,j in enumerate(OverallSyst.u_names) if 'w_rail' in j]])
# # plt.plot(x,u[[i for i,j in enumerate(OverallSyst.u_names) if 'w_rail' in j]],'x')
# =============================================================================