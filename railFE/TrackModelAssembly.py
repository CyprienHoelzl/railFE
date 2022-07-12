# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:21:46 2020

Track Model assembly

@author: CyprienHoelzl
"""
import numpy as np
from railFE.TimoshenkoBeamModel import Timoshenko4,Timoshenko4eb
from railFE.MatrixAssemblyOperations import addAtPos, addAtIdx
#%% Track Components    
class UIC60properties():
    def __init__(self):
        # Initialize Rail with UIC60 properties as described in:
        # http://www.railway-research.org/IMG/pdf/319-2.pdf
        self.k = 0.38 # [] UIC 60 < 1500Hz, k=0.38 if >1500Hz
        self.E = 210*10**9 # [Pa] (210GPA)
        self.A = 76.9*0.01**2 # [m**2]
        self.G = 77*10**9 # [Pa]
        self.I = 3055*0.01**4# [m^4]
        self.I_xx = 513*0.01**4# [m^4]
        self.I_yy = 3055*0.01**4# [m^4]
        self.I_xy = 0
        self.rho = 7850 # [kg/m^3]
        self.K_c0 = 1.028*10**9 #[N/m] Hertzian Spring Stiffness
        self.K_c = 100.0*10**9 #[N/m^1.5] Hertzian Spring coefficient
    def Psi(self,L):
        # Defined as: Psi = 12EI/GAkL^2
        return 12*self.E*self.I/(self.G*self.A*self.k*L**2)
    def I_rotated(self):
        # Inertia defined as:
        angle = np.arcsin(1/20)
        I_xx = self.I_xx
        I_yy = self.I_yy
        I_xy = self.I_xy
        I_uu = (I_xx + I_yy)/2 + (I_xx-I_yy)*np.cos(2*angle)-I_xy*np.sin(2*angle)
        I_vv = (I_xx + I_yy)/2 - (I_xx-I_yy)*np.cos(2*angle)+I_xy*np.sin(2*angle)
        I_uv = (I_xx-I_yy)*np.sin(2*angle)+I_xy*np.cos(2*angle)
        return I_uu,I_vv,I_uv
        
class SleeperB70():
    def __init__(self):
        # Initialization of sleeper parameters for a B70 type sleeper:
        # https://www.railone.de/produkte-loesungen/fern-gueterverkehr/schotteroberbau/betonschwelle-b70
        self.m_s_half = 140 #kg
        self.m_s = 280 #kg
        self.I = np.infty # Sleeper elasticity neglected
class SleeperB90():
    def __init__(self):
        # Initialization of sleeper parameters for a B90 type sleeper:
        # https://www.railone.de/produkte-loesungen/fern-gueterverkehr/schotteroberbau/betonschwelle-b70
        self.m_s_half = 177.5# 140 #kg
        self.m_s = 355 #kg
        self.I = np.infty # Sleeper elasticity neglected
class Ballast():
    def __init__(self,K=40*10**6,C=47*10**3):
        # Initialization of ballast parameters
        self.K_b = K # [N/m] Ballast Stiffness
        # self.K_b = 1/(1/(70*10**6)+1/(207*1000*250)) # USP http://www.trackelast.com/usp7ms2025.html
        self.C_b = C # [Ns/m] Ballast Damping
class Pad():
    def __init__(self,K =  200*10**6, C=28*10**3):#48*10**3):
        # Initialization of pad parameters
        self.K_p =K # [N/m] Pad Stiffness
        self.C_p =C # [Ns/m] Pad Damping
    def distributed_params(self,L_s):
        self.k_pd = self.K_p/L_s
        self.c_pd = self.C_p/L_s        
#%% Track Assembly
class TrackAssembly():
    def __init__(self,support_type = 'eb', n_sleepers = 81,sleeper_spacing = 0.6,support_length = 0.16,**kwargs):
        """
        Initialize with the default track assembly
        
        -----------------   rail
          p    p    p       pad
          s    s    s       sleeper
          b    b    b       ballast
          
          
        Parameters
        ----------
        support_type : 'pt' point support or 'eb' elastic base. The default is 'pt'.
        n_sleepers : number of sleepers. The default is 81.
        sleeper_spacing : TYPE, optional
            DESCRIPTION. The default is 0.6.
        support_length : TYPE, optional
            DESCRIPTION. The default is 0.16.

        Returns
        -------
        None.

        """
        assert (n_sleepers % 2)!=0, "The number of sleepers must be odd"
        assert (support_type in ['pt','eb']), "Track types: eb (distributed pad support) or pt (point pad support)"
        self.n_sleepers      = n_sleepers # Number of sleepers in FE-Model
        self.sleeper_spacing = sleeper_spacing 
        self.support_length  = support_length
        # Sleepers and Ballast
        self.SleeperB90 = SleeperB90()
        self.Ballast = Ballast()
        self.Pad = Pad()
        self.modal_analysis = True
        for k,v in kwargs.items():
            if hasattr(self,k):
                setattr(self,k,v)
            else:
                raise Warning('Unknown attribute{}'.format(str(k)))
        # Track Fe-Matrices
        if support_type == 'eb':
            # Rail elements
            self.Timoshenko4    = Timoshenko4(UIC60properties(),(self.sleeper_spacing-self.support_length)/2,modal_analysis  = self.modal_analysis)
            self.Timoshenko4eb = Timoshenko4eb(UIC60properties(),self.Pad,self.support_length,modal_analysis  = self.modal_analysis)
            # Assembling the systems of equations
            self.assembleTrackMatricesEB3el()
        elif support_type == 'pt':
            # Rail elements
            self.Timoshenko4    = Timoshenko4(UIC60properties(),(self.sleeper_spacing)/4)
            # Assembling the systems of equations            
            self.assembleTrackMatricesPT4el()

    def assembleTrackMatricesEB3el(self):
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
        
        # Definition of M,C,K Matrices for TIM4 Element
        tim4M = self.Timoshenko4.massMatrix()
        tim4K = self.Timoshenko4.stiffnessMatrix()
        tim4C = self.Timoshenko4.dampingMatrix()  
        # Definition of M,C,K Matrices for TIM4eb element
        tim4elM = self.Timoshenko4eb.massMatrixTEEF()
        tim4elK = self.Timoshenko4eb.stiffnessMatrixTEEF()
        tim4elC = self.Timoshenko4eb.dampingMatrixTEEF()
        # Definition of sleeper and Ballast Params
        M_sleeper         = self.SleeperB90.m_s_half
        K_sleeper_ballast = self.Ballast.K_b
        C_sleeper_ballast = self.Ballast.C_b
        print('Warning: start and end on rail!!')
        # Base 3 element assembly TIM4el (5nodes) + left/right TIM4 (2*2nodes)
        M_3elems = addAtIdx(addAtPos(addAtPos(np.zeros((9,9)),tim4M,(0,0)),tim4elM,(2,2)),tim4M,([4,5,7,8],[4,5,7,8]))
        K_3elems = addAtIdx(addAtPos(addAtPos(np.zeros((9,9)),tim4K,(0,0)),tim4elK,(2,2)),tim4K,([4,5,7,8],[4,5,7,8]))
        C_3elems = addAtIdx(addAtPos(addAtPos(np.zeros((9,9)),tim4C,(0,0)),tim4elC,(2,2)),tim4C,([4,5,7,8],[4,5,7,8]))
        
        # Assembly of Based 3 elements
        M_track = np.zeros((self.n_sleepers*7+2,self.n_sleepers*7+2))
        K_track = np.zeros((self.n_sleepers*7+2,self.n_sleepers*7+2))
        C_track = np.zeros((self.n_sleepers*7+2,self.n_sleepers*7+2))
        for i in range(self.n_sleepers):
            M_track = addAtPos(M_track,M_3elems,(7*i,7*i))
            K_track = addAtPos(K_track,K_3elems,(7*i,7*i))
            C_track = addAtPos(C_track,C_3elems,(7*i,7*i))
            # Connect the sleeper to ground stiffness, add sleeper mass
            M_track = addAtIdx(M_track, M_sleeper, ([6+7*i],[6+7*i]))
            K_track = addAtIdx(K_track, K_sleeper_ballast, ([6+7*i],[6+7*i]))
            C_track = addAtIdx(C_track, C_sleeper_ballast, ([6+7*i],[6+7*i]))
        
        # Setting the objects
        self.M_tr = M_track 
        self.K_tr = K_track 
        self.C_tr = C_track       
        self.u_names = ([['w_rail_{}_3'.format(str(0)),'t_rail_{}_3'.format(str(0))]]+
                        [['w_rail_{}_1'.format(str(i+1)),'t_rail_{}_1'.format(str(i+1)),
                         'w_rail_{}_2'.format(str(i+1)),'t_rail_{}_2'.format(str(i+1)),
                         'w_sleeper_{}_1'.format(str(i+1)),
                         'w_rail_{}_3'.format(str(i+1)),'t_rail_{}_3'.format(str(i+1))] for i in range(self.n_sleepers)])
        self.u_names_l = [j for i in self.u_names for j in i]
    def assembleTrackMatricesPT4el(self):
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
        n_nodes = 4
        n_el_p_bay = n_nodes*2+1
        # Definition of M,C,K Matrices for TIM4 Element
        tim4M = self.Timoshenko4.massMatrix()
        tim4K = self.Timoshenko4.stiffnessMatrix()
        tim4C = self.Timoshenko4.dampingMatrix()  
        # Definition of sleeper and Ballast Params
        M_sleeper         = self.SleeperB90.m_s_half
        K_sleeper_ballast = self.Ballast.K_b
        C_sleeper_ballast = self.Ballast.C_b
        
        K_s = np.array([[self.Pad.K_p, -self.Pad.K_p],
                        [-self.Pad.K_p, self.Pad.K_p+K_sleeper_ballast]])
        C_s = np.array([[self.Pad.C_p, -self.Pad.C_p],
                        [-self.Pad.C_p, self.Pad.C_p+C_sleeper_ballast]])
        # Base 4 element assembly TIM4el (5nodes) + left/right TIM4 (2*2nodes)
        M_4elems = addAtPos(addAtIdx(addAtIdx(addAtPos(addAtPos(np.zeros((n_el_p_bay+2,n_el_p_bay+2)),tim4M,(0,0)),tim4M,(2,2)),
                                              [M_sleeper],([6],[6])),
                                     tim4M,([4,5,7,8],[4,5,7,8])),tim4M,(7,7))
        K_4elems = addAtPos(addAtIdx(addAtIdx(addAtPos(addAtPos(np.zeros((n_el_p_bay+2,n_el_p_bay+2)),tim4K,(0,0)),tim4K,(2,2)),
                                              K_s, ([4,6],[4,6])), 
                                     tim4K,([4,5,7,8],[4,5,7,8])),tim4K,(7,7))
        C_4elems = addAtPos(addAtIdx(addAtIdx(addAtPos(addAtPos(np.zeros((n_el_p_bay+2,n_el_p_bay+2)),tim4C,(0,0)),tim4C,(2,2)),
                                              C_s, ([4,6],[4,6])),
                                     tim4C,([4,5,7,8],[4,5,7,8])),tim4C,(7,7))
        K_4elems[5,5] = K_4elems[5,5] #+ self.Pad.K_p*0.15**2/12
        C_4elems[5,5] = C_4elems[5,5] #+ self.Pad.C_p*0.15**2/12
        # Assembly of Based 3 elements
        M_track = np.zeros(((self.n_sleepers-2)*n_el_p_bay+2+10,(self.n_sleepers-2)*n_el_p_bay+2+10))
        K_track = np.zeros(((self.n_sleepers-2)*n_el_p_bay+2+10,(self.n_sleepers-2)*n_el_p_bay+2+10))
        C_track = np.zeros(((self.n_sleepers-2)*n_el_p_bay+2+10,(self.n_sleepers-2)*n_el_p_bay+2+10))
        for i in range(self.n_sleepers-2):
            M_track = addAtPos(M_track,M_4elems,(n_el_p_bay*i+5,n_el_p_bay*i+5))
            K_track = addAtPos(K_track,K_4elems,(n_el_p_bay*i+5,n_el_p_bay*i+5))
            C_track = addAtPos(C_track,C_4elems,(n_el_p_bay*i+5,n_el_p_bay*i+5))
        
        # Boundaries are modelled as supports with half stiffness and half damping
        M_track[:7,:7] = M_track[:7,:7] + M_4elems[4:,4:]
        K_track[:7,:7] = K_track[:7,:7] + K_4elems[4:,4:]
        C_track[:7,:7] = C_track[:7,:7] + C_4elems[4:,4:]
        K_track[2,2] = K_track[2,2] - K_sleeper_ballast/2
        C_track[2,2] = C_track[2,2] - C_sleeper_ballast/2
        M_track[-7:,-7:] = M_track[-7:,-7:] + M_4elems[:7,:7]
        K_track[-7:,-7:] = K_track[-7:,-7:] + K_4elems[:7,:7]
        C_track[-7:,-7:] = C_track[-7:,-7:] + C_4elems[:7,:7]
        K_track[-1,-1] = K_track[-1,-1] - K_sleeper_ballast/2
        C_track[-1,-1] = C_track[-1,-1] - C_sleeper_ballast/2
        # Setting the objects
        self.M_tr = M_track 
        self.K_tr = K_track 
        self.C_tr = C_track 
        self.u_names = ([['w_rail_{}_2'.format(str(0)),'t_rail_{}_2'.format(str(0)),
                         'w_sleeper_{}_1'.format(str(0)),
                         'w_rail_{}_3'.format(str(0)),'t_rail_{}_3'.format(str(0)),
                         'w_rail_{}_4'.format(str(0)),'t_rail_{}_4'.format(str(0))]]+
                        [['w_rail_{}_1'.format(str(i+1)),'t_rail_{}_1'.format(str(i+1)),
                         'w_rail_{}_2'.format(str(i+1)),'t_rail_{}_2'.format(str(i+1)),
                         'w_sleeper_{}_1'.format(str(i+1)),
                         'w_rail_{}_3'.format(str(i+1)),'t_rail_{}_3'.format(str(i+1)),
                         'w_rail_{}_4'.format(str(i+1)),'t_rail_{}_4'.format(str(i+1))] for i in range(self.n_sleepers-2)]+
                        [['w_rail_{}_1'.format(str(self.n_sleepers-1)),'t_rail_{}_1'.format(str(self.n_sleepers-1)),
                          'w_rail_{}_2'.format(str(self.n_sleepers-1)),'t_rail_{}_2'.format(str(self.n_sleepers-1)),
                         'w_sleeper_{}_1'.format(str(self.n_sleepers-1))]])
        self.u_names_l = [j for i in self.u_names for j in i]
        
        
if __name__ == "__main__":  
    #%%% Testing System Matrix Assembly track
    Track = TrackAssembly()
    # # Definition of M,C,K Matrices for TIM4 Element
    # tim4 = Track.Timoshenko4
    # tim4M = tim4.massMatrix()
    # tim4K = tim4.stiffnessMatrix()
    # tim4C = tim4.dampingMatrix() #np.zeros((4,4))
    # # Definition of M,C,K Matrices for TIM4eb element
    # tim4el = Track.Timoshenko4eb
    # tim4elM = tim4el.massMatrixTEEF()
    # tim4elK = tim4el.stiffnessMatrixTEEF()
    # tim4elC = tim4el.dampingMatrixTEEF()
    
    # # Base 3 element assembly TIM4el (5nodes) + left/right TIM4 (2*2nodes)
    # M_3elems = addAtIdx(addAtPos(addAtPos(np.zeros((9,9)),tim4M,(0,0)),tim4elM,(2,2)),tim4M,([4,5,7,8],[4,5,7,8]))
    # K_3elems = addAtIdx(addAtPos(addAtPos(np.zeros((9,9)),tim4K,(0,0)),tim4elK,(2,2)),tim4K,([4,5,7,8],[4,5,7,8]))
    # C_3elems = addAtIdx(addAtPos(addAtPos(np.zeros((9,9)),tim4C,(0,0)),tim4elC,(2,2)),tim4C,([4,5,7,8],[4,5,7,8]))
    
    # # Assembly of Based 3 elements
    # M_track = np.zeros((Track.n_sleepers*7+2,Track.n_sleepers*7+2))
    # K_track = np.zeros((Track.n_sleepers*7+2,Track.n_sleepers*7+2))
    # C_track = np.zeros((Track.n_sleepers*7+2,Track.n_sleepers*7+2))
    # for i in range(Track.n_sleepers):
    #     M_track = addAtPos(M_track,M_3elems,(7*i,7*i))
    #     K_track = addAtPos(K_track,K_3elems,(7*i,7*i))
    #     C_track = addAtPos(C_track,C_3elems,(7*i,7*i))
    # # Corresponding DOF names
    # u_nameslist = Track.u_names_l