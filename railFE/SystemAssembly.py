# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:21:46 2020

Matrix Assembly of System

@author: CyprienHoelzl
"""
import numpy as np
import scipy
import control
from railFE.VehicleModelAssembly import VehicleAssembly
from railFE.TrackModelAssembly import TrackAssembly   
from railFE.MatrixAssemblyOperations import addAtPos, addAtIdx 
#%% Local System Assembly
class LocalAssembly():
    def __init__(self, OverallSystem,modal_analysis = True, modal_bays = 5):
        self.OverallSystem = OverallSystem
        self.modal_analysis = modal_analysis
        self.modal_bays = modal_bays
        if self.modal_analysis:      
            modal_n_modes = self.OverallSystem.Track.Timoshenko4.modal_n_modes
            mid_sleeper = int((self.OverallSystem.n_sleepers-1)/2)
            sleepers = list(range((mid_sleeper-int((self.modal_bays-1)/2)-1),(mid_sleeper+int((self.modal_bays-1)/2)+1)))
            if self.OverallSystem.support_type == 'eb':
                nodes    = [12,23,31]
                self.assembleLocalMatricesEB3el()
            elif self.OverallSystem.support_type == 'pt':
                nodes    = [12,23,34,41]
                self.assembleLocalMatricesPT4el()
            modes    = list(range(modal_n_modes))
            u_names=[]
            for sleeper in sleepers:
                for node in nodes:
                    for mode in modes:
                        u_names.append('w_modal_{sleeper}_{node}_{mode}'.format(**{'sleeper':sleeper,'node':node,'mode':mode+1}))
            self.u_names_l = u_names[(len(nodes)-1)*modal_n_modes:-modal_n_modes]
        else:
            self.u_names_l = []
        self.K_loc = np.zeros((len(self.u_names_l)+OverallSystem.Track.K_tr.shape[0]+OverallSystem.Vehicle.K_veh.shape[0],
                               len(self.u_names_l)+OverallSystem.Track.K_tr.shape[0]+OverallSystem.Vehicle.K_veh.shape[0]))
        self.f_loc = np.zeros((len(self.u_names_l)+OverallSystem.Track.K_tr.shape[0]+OverallSystem.Vehicle.K_veh.shape[0],1))
        self.f_int = np.zeros((len(self.u_names_l)+OverallSystem.Track.K_tr.shape[0]+OverallSystem.Vehicle.K_veh.shape[0],1))    
    
    def assembleLocalMatricesEB3el(self):
        # Summary printout
        """Assembling System Matrix for 4DOF Timoshenko elements and 4DOF elastically supported Timoshenko element')
        'The model starts with sleeper 0 and end with sleeper {}'.format(str(self.n_sleepers-1)))
        Numbering convention:  w_i_1, t_i_1, w_i_s, w_i_2, t_i_2, w_i_3, t_i_3
        Where: 
              \tw is the vertical displacement, 
              \tt is the rotation theta, 
              \ti is the number of bay element, 
              \tnode _1 and _2 are on the left/right sleeper_i side respectively,
              \tnode _s is the sleeper node,
              \tnode _3 is the mid span node."""
        
        no_modes = self.OverallSystem.Track.Timoshenko4.modal_n_modes
        n_nodes = 3
        n_el_p_bay = n_nodes*2+1
        
        # Definition of M,C,K Matrices for TIM4 Element
        timMcross = self.OverallSystem.Track.Timoshenko4.M_cross_dyn().T
        timM = self.OverallSystem.Track.Timoshenko4.M_loc_dyn()
        timK = self.OverallSystem.Track.Timoshenko4.K_loc_dyn()
        timC = self.OverallSystem.Track.Timoshenko4.C_loc_dyn() 
        # Definition of M,C,K Matrices for TIM4eb element
        timelMcross = self.OverallSystem.Track.Timoshenko4eb.M_cross_dyn().T
        timelM = self.OverallSystem.Track.Timoshenko4eb.M_loc_dyn()
        timelK = self.OverallSystem.Track.Timoshenko4eb.K_loc_dyn()
        timelC = self.OverallSystem.Track.Timoshenko4eb.C_loc_dyn()
        
        # Base 3 element assembly TIM4el (5nodes) + left/right TIM4 (2*2nodes)    
        M_cross3elems = addAtIdx(addAtPos(addAtPos(np.zeros((n_el_p_bay+2,n_nodes*no_modes)),timMcross,(0,0)),
                                          timelMcross,(no_modes,2)),
                                     timMcross,(list(no_modes*2+np.arange(no_modes)),[4,5,7,8]))
        M_3elems = scipy.linalg.block_diag(timM,timelM,timM)
        K_3elems = scipy.linalg.block_diag(timK,timelK,timK)
        C_3elems = scipy.linalg.block_diag(timC,timelC,timC)
        
        # Assembly of Based 3 elements
        Mcross_local = np.zeros((self.modal_bays*n_el_p_bay+2,self.modal_bays*n_nodes*no_modes))
        M_local      = np.zeros((self.modal_bays*n_nodes*no_modes,self.modal_bays*n_nodes*no_modes))
        K_local      = np.zeros((self.modal_bays*n_nodes*no_modes,self.modal_bays*n_nodes*no_modes))
        C_local      = np.zeros((self.modal_bays*n_nodes*no_modes,self.modal_bays*n_nodes*no_modes))
        for i in range(self.modal_bays):
            Mcross_local = addAtPos(Mcross_local,M_cross3elems,(i*n_nodes*no_modes,n_el_p_bay*i))
            M_local = addAtPos(M_local,M_3elems,(i*n_nodes*no_modes,i*n_nodes*no_modes))
            K_local = addAtPos(K_local,K_3elems,(i*n_nodes*no_modes,i*n_nodes*no_modes))
            C_local = addAtPos(C_local,C_3elems,(i*n_nodes*no_modes,i*n_nodes*no_modes))
        
        x1,x2 = (int(n_el_p_bay*(self.OverallSystem.n_sleepers-self.modal_bays-2)/2)), (int(n_el_p_bay*(self.OverallSystem.n_sleepers-self.modal_bays+2)/2))
        Mcross_local = np.vstack((np.zeros((x1,self.modal_bays*n_nodes*no_modes)),Mcross_local,np.zeros((x2,self.modal_bays*n_nodes*no_modes))))        
        # Setting the objects
        self.Mcross_local = Mcross_local
        self.M_local = M_local 
        self.K_local = K_local 
        self.C_local = C_local 
    def assembleLocalMatricesPT4el(self):
        # Summary printout
        """Assembling System Matrix for 4DOF Timoshenko elements with point support
        'The model starts with sleeper 0 and end with sleeper {}'.format(str(self.n_sleepers-1)))
        Numbering convention:  w_rail_i_1, t_rail_i_1, w_rail_i_2, t_rail_i_2, w_rail_i_s, w_rail_i_3, t_rail_i_3, w_rail_i_4, t_rail_i_4
        Where: 
              \tw is the vertical displacement, 
              \tt is the rotation theta, 
              \ti is the number of bay element, 
              \tnode _1 and _3 are on the left/right sleeper_i side respectively,
              \tnode _2 are is the rail node connected to the track,
              \tnode _s is the sleeper node,
              \tnode _4 is the mid span node.
        """
        no_modes = self.OverallSystem.Track.Timoshenko4.modal_n_modes
        n_nodes = 4
        n_el_p_bay = n_nodes*2+1
        # Definition of M,C,K Matrices for TIM4 Element
        tim4Mcross = self.OverallSystem.Track.Timoshenko4.M_cross_dyn().T
        tim4M = self.OverallSystem.Track.Timoshenko4.M_loc_dyn()
        tim4K = self.OverallSystem.Track.Timoshenko4.K_loc_dyn()
        tim4C = self.OverallSystem.Track.Timoshenko4.C_loc_dyn()
        # Base 4 element assembly left/right TIM4 (2*2nodes)
        M_cross4elems = addAtPos(addAtIdx(addAtPos(addAtPos(np.zeros((n_el_p_bay+2,n_nodes*no_modes)),tim4Mcross,(0,0)),tim4Mcross,(no_modes,2)),
                                     tim4Mcross,(list(no_modes*2+np.arange(no_modes)),[4,5,7,8])),tim4Mcross,(no_modes*3,7))
        M_4elems = scipy.linalg.block_diag(tim4M,tim4M,tim4M,tim4M)
        K_4elems = scipy.linalg.block_diag(tim4K,tim4K,tim4K,tim4K)
        C_4elems = scipy.linalg.block_diag(tim4C,tim4C,tim4C,tim4C)

        # Assembly of Based 4 elements
        Mcross_local = np.zeros((self.modal_bays*n_el_p_bay+2,self.modal_bays*n_nodes*no_modes))
        M_local      = np.zeros((self.modal_bays*n_nodes*no_modes,self.modal_bays*n_nodes*no_modes))
        K_local      = np.zeros((self.modal_bays*n_nodes*no_modes,self.modal_bays*n_nodes*no_modes))
        C_local      = np.zeros((self.modal_bays*n_nodes*no_modes,self.modal_bays*n_nodes*no_modes))
        for i in range(self.modal_bays):
            Mcross_local = addAtPos(Mcross_local,M_cross4elems,(i*n_nodes*no_modes,n_el_p_bay*i))
            M_local = addAtPos(M_local,M_4elems,(i*n_nodes*no_modes,i*n_nodes*no_modes))
            K_local = addAtPos(K_local,K_4elems,(i*n_nodes*no_modes,i*n_nodes*no_modes))
            C_local = addAtPos(C_local,C_4elems,(i*n_nodes*no_modes,i*n_nodes*no_modes))
        x1,x2 = (5+ int(n_el_p_bay*(self.OverallSystem.n_sleepers-self.modal_bays-2)/2)), (5+ int(n_el_p_bay*(self.OverallSystem.n_sleepers-self.modal_bays-2)/2))
        Mcross_local = np.vstack((np.zeros((x1,self.modal_bays*n_nodes*no_modes)),Mcross_local,np.zeros((x2,self.modal_bays*n_nodes*no_modes))))
        # Setting the objects
        self.Mcross_local = Mcross_local
        self.M_local = M_local 
        self.K_local = K_local 
        self.C_local = C_local      
    def interpolation_matrix(self,xi,node_type = 'TIM4'): 
        if node_type == 'TIM4EB':
            N_0 = self.OverallSystem.Track.Timoshenko4eb.N_EF_w(xi,0)
            N_1 = self.OverallSystem.Track.Timoshenko4eb.N_EF_w(xi,1)
            N_2 = self.OverallSystem.Track.Timoshenko4eb.N_EF_w(xi,2)
            N_3 = self.OverallSystem.Track.Timoshenko4eb.N_EF_w(xi,3)
            Psi = self.OverallSystem.Track.Timoshenko4eb.Psi_Modal(xi)  
        elif node_type == 'TIM4':
            N_0 = self.OverallSystem.Track.Timoshenko4.N_w1(xi)
            N_1 = self.OverallSystem.Track.Timoshenko4.N_t1(xi)
            N_2 = self.OverallSystem.Track.Timoshenko4.N_w2(xi)
            N_3 = self.OverallSystem.Track.Timoshenko4.N_t2(xi) 
            Psi = self.OverallSystem.Track.Timoshenko4.Psi_Modal(xi)           
        if self.modal_analysis == True:
            # if self.OverallSystem.support_type == 'eb':
            #     Psi_eb = OverallSyst.Track.Timoshenko4eb.Psi_Modal(xi)  
            #     Psi_pt = OverallSyst.Track.Timoshenko4.Psi_Modal(xi)        
            #     Psi_overall=[]
            #     for i in range(self.modal_bays):
            #         Psi_overall+=Psi_pt+Psi_eb+Psi_pt
            # elif self.OverallSystem.support_type == 'pt':
            #     Psi_pt = OverallSyst.Track.Timoshenko4.Psi_Modal(xi)               
            #     Psi_overall=[]
            #     for i in range(self.modal_bays):
            #         Psi_overall+=Psi_pt+Psi_pt+Psi_pt+Psi_pt
            self.E = np.dot(np.array([[1,-N_0,-N_1,-N_2,-N_3]+ (-np.array(Psi)).tolist() +[-1]]).T,np.array([[-1,N_0,N_1,N_2,N_3]+Psi+[1]]))   
        else:
            self.E = np.dot(np.array([[1,-N_0,-N_1,-N_2,-N_3,-1]]).T,np.array([[-1,N_0,N_1,N_2,N_3,1]])) 
        return self.E

    def assembleLocalMatrices(self,xi,segment,K_c,Yi,w_irr):
        '''The local system connect the vehicle to the track with a non-linear Hertzian spring
        The force from the contact on the system's DOFs:
        f_c = -[-1, N, 1].T*f_c = -[-1, N, 1].T*K_c*delta**1.5
        where delta = w_G,r + w_loc + w_irr - w_w
        K_c here assumed constant
          w_G,r=N(x_w)*q_tr and w_loc=q_l refer to global and local rail displacement respectively,
          w_w is the wheel displacement
          w_irr are wheel-rail body contact irregularities
        Vectorial form:
        f_c = -[-1, N, 1].T*f_c = -K_c*delta**0.5*[-1, N, 1].T*[1, N, 1, -1]*[w_w,q_tr,q_L,w_irr].T
        Separating irregularities:
        f_c = K_c*delta**0.5*[1, -N, -1].T*[-1, N, 1]*[w_w,q_tr,q_L].T +  K_c*delta**0.5*[1, -N, -1].T*w_irr
        f_c = K_c*delta**0.5*E*q_sys + f_irr
        
        [K_sys - K_c*delta**0.5]*q_sys
        where delta evaluated during convergence process.
        and 
        E is a function of N(xi_wheel)
        '''
        K_c = K_c*1.0
        idxlist = [np.where(np.array(self.OverallSystem.u_names)=='w_axle_1')[0][0]]+ segment['node_indexes'] + segment['modal_indexes']
        E = self.interpolation_matrix(xi,segment['type']) 
        if self.OverallSystem.modal_analysis:
            # print(segment['type'])
            if segment['type'] == 'TIM4EB':
                track_flexibility = self.OverallSystem.Track.Timoshenko4eb.point_load_flexibility_TIM4(xi)
                track_flexibility = self.OverallSystem.Track.Timoshenko4eb.F_res(xi)
            elif segment['type'] == 'TIM4':
                track_flexibility = self.OverallSystem.Track.Timoshenko4.point_load_flexibility_TIM4(xi)
                track_flexibility = self.OverallSystem.Track.Timoshenko4.F_res(xi)
        else:
            if segment['type'] == 'TIM4EB':
                track_flexibility = self.OverallSystem.Track.Timoshenko4eb.point_load_flexibility_TIM4(xi)
            elif segment['type'] == 'TIM4':
                track_flexibility = self.OverallSystem.Track.Timoshenko4.point_load_flexibility_TIM4(xi)
        if hasattr(self,'K_loc_mod'):
            # if (segment['id']==self.OverallSystem.repeated_segments_coordinates['id'][0] and 
            #     self.OverallSystem.previous_segment==self.OverallSystem.repeated_segments_coordinates['id'][-1]):
            #     # If returning to previous state
            #     self.K_loc_mod = addAtIdx(np.copy(self.K_loc),self.K_fc[:-1,:-1],(idxlist,idxlist))
            if (segment['id']!= self.OverallSystem.previous_segment):
                F_wheelrail=self.F_wheelrail
            else:
                F_wheelrail =np.dot(self.OverallSystem.K_sys[1,:]-self.K_loc_mod[1,:],Yi[:self.OverallSystem.n_dof])
        else:
            F_wheelrail = np.dot(self.OverallSystem.K_sys[1,:],Yi[:self.OverallSystem.n_dof])
        self.track_flexibility=track_flexibility
        self.F_wheelrail=F_wheelrail
        self.u_rail = track_flexibility*F_wheelrail
        self.Yi = Yi
        # print('todo')
        delta = np.dot(E[:-1,0],Yi[idxlist]) + E[-1,0]*self.u_rail
       
        if delta<=0:
            print(xi,segment['type'])
            print('additional rail displacement under current load',self.u_rail)
            print('negative delta:', delta)
            delta = delta *0
            #self.delta= delta 
            delta = self.delta
        else:
            self.delta= delta
        # else:
        #      print( K_c*delta**(1/2)*E[:,-1].reshape((-1,1))*self.u_rail)
        # Stiffness of contact spring
        Kc =self.OverallSystem.Track.Timoshenko4.railproperties.K_c0 #  K_c*delta**(1/2) #self.OverallSystem.Track.Timoshenko4.railproperties.K_c0 #
        self.K_fc = Kc*E
        # print(Kc/self.OverallSystem.Track.Timoshenko4.railproperties.K_c0)
        # if abs(self.K_fc[0,0]/10**9-0.54)>0.1:
        #     print('a',self.K_fc[0,0]/10**9)
        # Force due to irregularities on wheel or track
        self.f_lc = Kc*E[:,0].reshape((-1,1))*w_irr 
        # Force due to rail deformation
        self.f_it = Kc*E[:,-1].reshape((-1,1))*self.u_rail
        
        #print('stiffness',K_c*delta**(1/2),'delta',delta,'yi',Yi[idxlist][0])
        
        # Adding to main matrices
        self.K_loc_mod = addAtIdx(np.copy(self.K_loc),self.K_fc[:-1,:-1],(idxlist,idxlist))
        self.f_loc_mod = addAtIdx(np.copy(self.f_loc),self.f_lc[:-1,:],([0],idxlist))
        self.f_int_mod = addAtIdx(np.copy(self.f_int),self.f_it[:-1,:],([0],idxlist))
        return  self.K_loc_mod,self.f_loc_mod,self.f_int_mod
    def assembleLocalMatricesStatic(self,xi,segment,K_c,w_irr):
        """The local system connect the vehicle to the track with a linear Hertzian spring"""
        idxlist = [np.where(np.array(self.OverallSystem.u_names)=='w_axle_1')[0][0]]+ segment['node_indexes'] + segment['modal_indexes']
        E = self.interpolation_matrix(xi,segment['type']) 
        # Stiffness of contact spring
        self.K_fc_static = K_c*E
        # Force due to irregularities on wheel or track
        self.f_lc_static = - K_c*E[:,0].reshape((-1,1))*w_irr
        self.K_loc_mod_static = addAtIdx(np.copy(self.K_loc),self.K_fc_static[:-1,:-1],(idxlist,idxlist))
        self.f_loc_mod_static = addAtIdx(np.copy(self.f_loc),self.f_lc_static[:-1,:],([0],idxlist))
        return  self.K_loc_mod_static,self.f_loc_mod_static

#%% Overall System Equation Assembly
class OverallSystem():
    # M_sys*q_sys''+C_sys*q_sys'+K_sys*q_sys = f = f_c + f_ext
    
    # K_loc*w_loc = f_c
    def __init__(self,support_type, n_sleepers= 81,modal_analysis = True,function_track_roughness=None,function_wheel_roughness=None,**kwargs):
        #print('to do> Local assembly, Overall system and mass application from vehicle, herzian stiffness')
        self.support_type = support_type
        self.n_sleepers = n_sleepers
        self.modal_analysis = modal_analysis
        self.modal_bays     = 3
        self.Vehicle = VehicleAssembly()  
        self.Track = TrackAssembly(support_type = self.support_type, n_sleepers = self.n_sleepers, **kwargs)
        self.Local = LocalAssembly(self,modal_analysis = self.modal_analysis, modal_bays = self.modal_bays)
        if self.modal_analysis:
            self.K_sys = scipy.linalg.block_diag(self.Vehicle.K_veh,self.Track.K_tr,self.Local.K_local)
            self.C_sys = scipy.linalg.block_diag(self.Vehicle.C_veh,self.Track.C_tr,self.Local.C_local)
            self.M_sys = scipy.linalg.block_diag(self.Vehicle.M_veh,np.hstack((np.vstack((self.Track.M_tr,self.Local.Mcross_local.T)),np.vstack((self.Local.Mcross_local,self.Local.M_local)))))
        else:
            self.K_sys = scipy.linalg.block_diag(self.Vehicle.K_veh,self.Track.K_tr)
            self.C_sys = scipy.linalg.block_diag(self.Vehicle.C_veh,self.Track.C_tr)
            self.M_sys = scipy.linalg.block_diag(self.Vehicle.M_veh,self.Track.M_tr)
        # Force array initialized
        self.n_dof = len(self.Vehicle.u_names + self.Track.u_names_l + self.Local.u_names_l)
        self.f_sys = self.ExternalExcitation()
        
        
        self.K_sys_upd = np.copy(self.K_sys)
        self.f_sys_upd = np.copy(self.f_sys)
        self.f_int = np.zeros(self.f_sys.shape)
        
        
        self.u_names = self.Vehicle.u_names + self.Track.u_names_l + self.Local.u_names_l
        # Definition of nodes that are repeated
        self.repeated_segments_coordinates,self.repeated_segments = self.iterated_nodes()
        self.previous_segment = list(self.repeated_segments.keys())[0]
                
        # Track Irregularities
        # self.w_irr_t = 0
        # # Wheel Irregularities
        # self.w_irr_w = 0
        
        # Time Invariant State Space 
        observed_dofs = [[np.where(np.array(self.u_names)=='w_axle_1')[0][0]],[286,288]]
        self.assemble_state_space(observed_dofs)
        
        self.function_track_roughness = function_track_roughness
        if function_track_roughness is None:# if no function_track_roughness is given, take default one
            self.function_track_roughness = self.function_track_roughness_default
        self.function_wheel_roughness = function_wheel_roughness
        if function_wheel_roughness is None:# if no function_wheel_roughness is given, take default one
            self.function_wheel_roughness = self.function_wheel_roughness_default
        
    def w_irr_t(self,ti):
        w_irr = self.function_track_roughness(ti)
        return w_irr
    def w_irr_w(self,ti):
        w_irr = self.function_wheel_roughness(ti) # Wheel Irregularities, factor of Pi*D
        return w_irr
    
    def function_track_roughness_default(self,ti):
        # Track Irregularities
        # By default a unit impulse at 0.25s
        if hasattr(self,'ij')==False:
            self.ij = 0
        w_irr = 0
        if ti >=0.25 and self.ij<1000:
            # print(ti)
            w_irr = (-scipy.signal.ricker(1000,30)[self.ij]/1000
                     +np.roll(-scipy.signal.ricker(1000,30)/1000,100)[self.ij]
                     -scipy.signal.unit_impulse(1000,'mid')[self.ij]/10000
                    -np.roll(scipy.signal.unit_impulse(1000,'mid'),2)[self.ij]/10000
                    -np.roll(scipy.signal.unit_impulse(1000,'mid'),1)[self.ij]/20000)
            self.ij +=1
        w_irr = w_irr + np.random.random()/500000
        return w_irr
    def function_wheel_roughness_default(self,ti):
        # Wheel Irregularities, factor of Pi*D
        # By default no wheel roughness
        w_irr = 0        
        return w_irr
        
    
    def ExternalExcitation(self):
        idx_mass_body = 0
        idx_mass_axle = 1
        f_sys = np.zeros((self.n_dof,1))
        self.f_ext = -np.array([[self.Vehicle.CarBody.M_car,self.Vehicle.Axle.M_axle]]).T*9.81
        f_sys = addAtIdx(f_sys,self.f_ext,([0],[idx_mass_body,idx_mass_axle]))    
        return f_sys
    def updateSystem(self,ti,Yi,speed):
        xi,segment = self.timeToLocalReference(ti,speed)
            
        K_c = self.Track.Timoshenko4.railproperties.K_c

        w_irr = self.w_irr_t(ti) + self.w_irr_w(ti)
        # current_node
        (K_loc_mod,f_loc_mod,f_H) = self.Local.assembleLocalMatrices(xi,segment,K_c,Yi,w_irr)
        self.K_sys_upd = self.K_sys - K_loc_mod
        self.f_sys_upd = self.f_sys + f_loc_mod   
        self.f_int = f_H
        # if ti >= 0.2 and ti<=0.005/speed+0.2:
        #     print(ti)
        
    def iterated_nodes(self):
        if self.support_type=='pt':
            idx_w_4p = np.arange(self.n_dof)[np.array(self.u_names) == 'w_rail_{}_4'.format(str(int((self.n_sleepers-1)/2-1)))][0]
            idx_w_1 = np.arange(self.n_dof)[np.array(self.u_names) == 'w_rail_{}_1'.format(str(int((self.n_sleepers-1)/2)))][0]
            idx_w_2 = np.arange(self.n_dof)[np.array(self.u_names) == 'w_rail_{}_2'.format(str(int((self.n_sleepers-1)/2)))][0]
            idx_w_3 = np.arange(self.n_dof)[np.array(self.u_names) == 'w_rail_{}_3'.format(str(int((self.n_sleepers-1)/2)))][0]
            idx_w_4 = np.arange(self.n_dof)[np.array(self.u_names) == 'w_rail_{}_4'.format(str(int((self.n_sleepers-1)/2)))][0]
            
            idx_w_modal_41 = list(np.arange(self.n_dof)[['w_modal_{}_41'.format(str(int((self.n_sleepers-1)/2-1))) in i for i in self.u_names]])
            idx_w_modal_12 = list(np.arange(self.n_dof)[['w_modal_{}_12'.format(str(int((self.n_sleepers-1)/2))) in i for i in self.u_names]])
            idx_w_modal_23 = list(np.arange(self.n_dof)[['w_modal_{}_23'.format(str(int((self.n_sleepers-1)/2))) in i for i in self.u_names]])
            idx_w_modal_34 = list(np.arange(self.n_dof)[['w_modal_{}_34'.format(str(int((self.n_sleepers-1)/2))) in i for i in self.u_names]])
            repeated_segments = {'41':{'id':'41','type':'TIM4','node_indexes':[idx_w_4p,idx_w_4p+1,idx_w_1,idx_w_1+1],'modal_indexes':idx_w_modal_41, 'length':0.15},
                            '12':{'id':'12','type':'TIM4','node_indexes':[idx_w_1,idx_w_1+1,idx_w_2,idx_w_2+1],'modal_indexes':idx_w_modal_12, 'length':0.15},
                            '23':{'id':'23','type':'TIM4','node_indexes':[idx_w_2,idx_w_2+1,idx_w_3,idx_w_3+1],'modal_indexes':idx_w_modal_23, 'length':0.15},
                            '34':{'id':'34','type':'TIM4','node_indexes':[idx_w_3,idx_w_3+1,idx_w_4,idx_w_4+1],'modal_indexes':idx_w_modal_34, 'length':0.15}}
            repeated_segments_coordinates = {'id':['41','12','23','34'],'coords':[0,0.15,0.3,0.45,0.6]}
        elif self.support_type =='eb':
            idx_w_3p = np.arange(self.n_dof)[np.array(self.u_names) == 'w_rail_{}_3'.format(str(int((self.n_sleepers-1)/2-1)))][0]
            idx_w_1 = np.arange(self.n_dof)[np.array(self.u_names) == 'w_rail_{}_1'.format(str(int((self.n_sleepers-1)/2)))][0]
            idx_w_2 = np.arange(self.n_dof)[np.array(self.u_names) == 'w_rail_{}_2'.format(str(int((self.n_sleepers-1)/2)))][0]
            idx_w_3 = np.arange(self.n_dof)[np.array(self.u_names) == 'w_rail_{}_3'.format(str(int((self.n_sleepers-1)/2)))][0]
            
            idx_w_modal_31 = np.arange(self.n_dof)[['w_modal_{}_31'.format(str(int((self.n_sleepers-1)/2-1))) in i for i in self.u_names]].tolist()
            idx_w_modal_12 = np.arange(self.n_dof)[['w_modal_{}_12'.format(str(int((self.n_sleepers-1)/2))) in i for i in self.u_names]].tolist()
            idx_w_modal_23 = np.arange(self.n_dof)[['w_modal_{}_23'.format(str(int((self.n_sleepers-1)/2))) in i for i in self.u_names]].tolist()
            
            repeated_segments = {'31':{'id':'31','type':'TIM4EB','node_indexes':[idx_w_3p,idx_w_3p+1,idx_w_1,idx_w_1+1],'modal_indexes':idx_w_modal_31, 'length':0.22},
                            '12':{'id':'12','type':'TIM4','node_indexes':[idx_w_1,idx_w_1+1,idx_w_2,idx_w_2+1],'modal_indexes':idx_w_modal_12, 'length':0.16},
                            '23':{'id':'23','type':'TIM4','node_indexes':[idx_w_2,idx_w_2+1,idx_w_3,idx_w_3+1],'modal_indexes':idx_w_modal_23, 'length':0.22}}
            repeated_segments_coordinates =  {'id':['31','12','23'],'coords':[0,0.22,0.38,0.6]}
        return repeated_segments_coordinates,repeated_segments
    def timeToLocalReference(self,t_glob,speed):
        # Calculate Global Position
        pos = self.timeToGlobalPos(t_glob,speed)
        # Calculate Local Position on 0.6m sleeper bays
        pos_local = pos-0.6*round(np.floor(pos/0.6))
        if pos_local<0:
            # print('Position <0, rounding up: {}'.format(str(pos_local)))
            pos_local = 0
        elif pos_local >0.6:
            # print('Position >0.6, rounding down: {}'.format(str(pos_local)))
            pos_local = 0.6
        # Find on which node segment we are
        #todo: this could be done better/faster
        # print('pos,t,posloc',pos,t_glob, pos_local)
        boolean_coords = np.arange(len(self.repeated_segments_coordinates['id']))[(np.array(self.repeated_segments_coordinates['coords'])[1:]>pos_local) & (np.array(self.repeated_segments_coordinates['coords'])[:-1]<=pos_local)][0]
        segment = self.repeated_segments[self.repeated_segments_coordinates['id'][boolean_coords]]
        start_pos = self.repeated_segments_coordinates['coords'][:-1][boolean_coords]
        xi = (pos_local - start_pos) /segment['length']  
        return xi,segment
                  
    def timeToGlobalPos(self,t_glob,speed):
        # Relative Position at t=0:
        pos_rel_0 = 0 # 0::starts on node 41
        self.speed = speed # [m/s] constant velocity defined here
        pos_glob = self.speed*t_glob + pos_rel_0
        return pos_glob
    def assemble_state_space(self,observed_dofs):
        # Assemble the base state space representation (time invariant)
        M = self.M_sys
        K = self.K_sys
        C = self.C_sys
        f = self.f_sys
        n = f.shape[0]
        k = f.shape[1]
        m = len(observed_dofs)
        # Assemble the state space system
        A = np.vstack((np.hstack((np.zeros((n,n)),np.eye(n))),
                        np.hstack((-scipy.linalg.solve(M,K),-scipy.linalg.solve(M,C)))))
        B = np.vstack((np.zeros((n,k)),np.dot(np.linalg.inv(M),f)))
        # Output y= Cx+Du
        C = np.zeros((m,2*n))
        for i in range(m):
            C[i,observed_dofs[i]]=1/len(observed_dofs[i])
        # C = np.vstack((np.zeros((int(n/2)-1,m)),np.eye(m),np.zeros((n+int(n/2)-10,m)))).T
        D = np.zeros((m,k))
        
        print('currently the force vector is not passed as an output')        
        self.sys_upd = control.ss(A,B,C,D)
        self.sys = control.ss(A,B,C,D)
        return self.sys
    # def update_state_space(self,ti,Yi):
    #     xi,segment = self.timeToLocalReference(ti)
    #     # Shift Xi back to previous if we are resetting of the initial segment
    #     # print('todo,!!\n!!!!!\n!!!!\n!!!')
    #     # Generate the updated stiffness matrix and force vector
    #     self.updateSystem(xi,segment,Yi)
    #     # a=time.time()
    #     A_upd = np.copy(self.sys.A)
    #     if (self.Local.K_loc_mod!=0).any():
    #         # Update the time variant components
    #         A_upd[self.n_dof:,:self.n_dof] = self.sys.A[self.n_dof:,:self.n_dof] + scipy.linalg.solve(self.M_sys, self.Local.K_loc_mod)
        
    #     B_upd = np.copy(self.sys.B)
    #     if (self.Local.f_loc_mod!=0).any():
    #         B_upd[self.n_dof:,:self.sys.B.shape[1]] = self.sys.B[self.n_dof:,:self.sys.B.shape[1]] + scipy.linalg.solve( self.M_sys, self.Local.f_loc_mod)
        
    #     C_upd = self.sys.C
    #     D_upd = self.sys.D
    #     self.sys_upd.A = A_upd
    #     self.sys_upd.B = B_upd
    #     self.sys_upd.C = D_upd
    #     self.sys_upd.D = D_upd
        # print(time.time()-a)
def system_matrix(support_type= 'eb',**kwargs): # eb or pt
    OverallSyst = OverallSystem(support_type = support_type,**kwargs)
    M = OverallSyst.M_sys
    K = OverallSyst.K_sys
    C = OverallSyst.C_sys
    f = OverallSyst.f_sys
    u_names = np.array(OverallSyst.u_names[:len(f)])
    n_sleepers = OverallSyst.n_sleepers
    
    n = len(f) # number of DOFs
    mid_sleeper = str(int((n_sleepers-1)/2-1))
    if support_type == 'eb':
        dof_rail_mid_span   = np.arange(n)[[True if (i == 'w_rail_{}_3'.format(mid_sleeper)) else False for i in u_names]]
        dof_rail_support    = [np.arange(n)[[True if (i == 'w_rail_{}_1'.format(mid_sleeper)) else False for i in u_names]],
                               np.arange(n)[[True if (i == 'w_rail_{}_2'.format(mid_sleeper)) else False for i in u_names]]]
        dof_sleeper         = np.arange(n)[[True if (i == 'w_sleeper_{}_1'.format(mid_sleeper)) else False for i in u_names]]
        dofs_rail           = np.arange(n)[[True if ('w_rail_' in i) else False for i in u_names]]
        dofs_sleeper        = np.arange(n)[[True if ('w_sleeper_' in i) else False for i in u_names]]
        
        f1 = np.copy(f)
        f2 = np.copy(f)
        # f3 = np.copy(f)
        f1[dof_rail_mid_span] = 1
        f2[dof_rail_support,:] = 0.5
        # f3[1]=1
        f = np.hstack((f1,f2))#,f3))
        
    elif support_type == 'pt':
        dof_rail_mid_span = np.arange(n)[[True if (i == 'w_rail_{}_4'.format(mid_sleeper)) else False for i in u_names]]
        dof_rail_support  = [np.arange(n)[[True if (i == 'w_rail_{}_2'.format(mid_sleeper)) else False for i in u_names]]]
        dof_sleeper         = np.arange(n)[[True if (i == 'w_sleeper_{}_1'.format(mid_sleeper)) else False for i in u_names]]
        dofs_rail           = np.arange(n)[[True if ('w_rail_' in i) else False for i in u_names]]
        dofs_sleeper        = np.arange(n)[[True if ('w_sleeper_' in i) else False for i in u_names]]
      
        f1 = np.copy(f)
        f2 = np.copy(f)
        f1[dof_rail_mid_span] = 1
        f2[dof_rail_support,:] = 1
        f = np.hstack((f1,f2))
    return OverallSyst,M,K,C,f, dof_rail_mid_span,dof_rail_support,dof_sleeper,dofs_rail,dofs_sleeper

def assemble_state_space(K,M,C,f,observed_dofs):
    n= len(f)
    k = f.shape[1]
    m = len(observed_dofs)
    # Assemble the state space system
    A = np.vstack((np.hstack((np.zeros((n,n)),np.eye(n))),
                    np.hstack((-scipy.linalg.solve(M,K),-scipy.linalg.solve(M,C)))))
    B = np.vstack((np.zeros((n,k)),np.dot(np.linalg.inv(M),f)))
    C = np.zeros((m,2*n))
    for i in range(m):
        C[i,observed_dofs[i]]=1/len(observed_dofs[i])
    # C = np.vstack((np.zeros((int(n/2)-1,m)),np.eye(m),np.zeros((n+int(n/2)-10,m)))).T
    D = np.zeros((m,k))
    print(B.shape,D.shape)
    sys = control.ss(A,B,C,D)
    return sys

        
if __name__ == "__main__":        
    xi = 0.5
    OverallSyst = OverallSystem('eb')  