# -*- coding: utf-8 -*-
"""Solve time integration of system

Usage:
    See examples
    
@author: 
    CyprienHoelzl
"""
import scipy
import numpy as np
def addAtIdx(mat1, mat2, xypos):
    """
    Add two arrays of different sizes in location, where the offset is defined by xy coordinates.

    Parameters
    ----------
    mat1 : base array.
    mat2 : array to be summed to mat1.
    xypos : tuple ([x,x2,x3...],[y1,y2,y3,..]) containing coordinates in mat1.

    Returns
    -------
    mat1 : added arrays.

    """
    x, y = xypos
    mat1[np.ix_(y, x)] += mat2
    return mat1

def newmark_nl(MODEL):
    """
    Non Linear Newmark algorithm for time integration of a dynamic system.
    Standard algorithm as described by: https://ethz.ch/content/dam/ethz/special-interest/baug/ibk/structural-mechanics-dam/education/femII/presentation_05_dynamics_v1_EC.pdf

    Parameters
    ----------
    MODEL : State Space Model

    Returns
    -------
    MODEL : State Space Model

    """
    nt = MODEL.nt;
    dt = MODEL.dt;
    MODEL.nt=0;
    #####################################
    # Newmark params
    gamma = 1/2;
    beta = 1/4;
    
    a1 = 1/(beta*dt**2);
    a2 = 1/(beta*dt);
    a3 = (1-2*beta)/(2*beta);
    a4 = gamma/(beta*dt);
    a5 = 1-gamma/beta;
    a6 = (1-gamma/(2*beta))*dt;
    
    #####################################
    # Model params
    # [ MODEL ] = assemble_nl( MODEL ); ** in my case my model is already assembled, and I update only what is neccessary
    # [ MODEL ] = apply_bc_nl( MODEL );
    
    neq = np.size(MODEL.u);
    
    C = MODEL.OverallSystem.C_sys;
    M = MODEL.OverallSystem.M_sys;
    
    a1M = a1*M;
    a4C = a4*C;
    # Initializing output variables
    MODEL.Hist = np.zeros((4,nt+2));
    MODEL.U = np.zeros((neq,nt+2));
    MODEL.V = np.zeros((neq,nt+2));
    MODEL.A = np.zeros((neq,nt+2));
    
    v0 = np.zeros((neq,1));
    
    f0 = MODEL.OverallSystem.f_sys * 1; #1 would be the time variant part f(t)
    MODEL.U[:,0] = MODEL.u[:,0]
    
    a0 = np.linalg.solve(M,(f0-np.dot(MODEL.OverallSystem.K_sys,MODEL.U[:,0].reshape((-1,1)))-np.dot(C,v0)))
    uk = MODEL.U[:,0].reshape((-1,1));
    uik = uk;
    
    MODEL.V[:,0] = v0[:,0];
    MODEL.A[:,0] = a0[:,0];
    
    vk = v0;
    ak = a0; 
    fnorm = np.linalg.norm(MODEL.OverallSystem.f_sys);
    tol=1e-4;
    maxit = 40;
    # 
    # iii=0
    # import matplotlib.pyplot as plt
    # plt.figure()
    for i in range(MODEL.U.shape[1]-1):
        # print(i)
        t = i*dt;
        # When jumping back to the first segment, the different arrays are shifted to the original position [ remove this part if not needed]
        xi,segment = MODEL.OverallSystem.timeToLocalReference(t,MODEL.speed)
        if (segment['id']==MODEL.OverallSystem.repeated_segments_coordinates['id'][0] and 
                MODEL.OverallSystem.previous_segment==MODEL.OverallSystem.repeated_segments_coordinates['id'][-1]):
            MODEL.u = MODEL.shift(MODEL.u)
            uk      = MODEL.u
            vk      = MODEL.shift(vk)
            ak      = MODEL.shift(ak)
            # MODEL.OverallSystem.updateSystem(0,MODEL.u,MODEL.speed) 
            idxlist = [np.where(np.array(MODEL.OverallSystem.u_names)=='w_axle_1')[0][0]] + segment['node_indexes']  + segment['modal_indexes']
            MODEL.OverallSystem.K_sys_upd = MODEL.OverallSystem.K_sys - addAtIdx(np.copy(MODEL.OverallSystem.Local.K_loc) ,
                                                    MODEL.OverallSystem.Local.K_fc[:-1,:-1],
                                                    (idxlist,idxlist))
            MODEL.OverallSystem.f_int = MODEL.shift(MODEL.OverallSystem.f_int)
        MODEL.OverallSystem.previous_segment= segment['id']
            
        # Force vector, here time invariant
        fi = MODEL.OverallSystem.f_sys_upd
        
        va1 = a2*vk+a3*ak;
        va2 = a5*vk+a6*ak;
        # Compute response force
        R = M.dot(-va1) + C.dot(va2) + MODEL.OverallSystem.K_sys_upd.dot(MODEL.u)  - MODEL.OverallSystem.f_int - fi ;
        
        # Initialize iteration params
        Rnorm = np.linalg.norm(R);
        nit=0;
        MODEL.nt = i;
        ui=uk;
        uik=ui-uk;
        # Iterate non linear until convergence/maxit
        while ((Rnorm>tol*fnorm) & (nit<=maxit)):  
            # MODEL.OverallSystem.K_sys_upd is the stiffness matrix with updated non-linear contact stiffness calculated with contact from the previous step
            Keff =a1M+a4C+MODEL.OverallSystem.K_sys_upd 
            # $\Delta u = K^{-1}\cdot-R$
            du = np.linalg.solve(Keff,(-R));
            ui = ui+du;
            
            MODEL.u = ui;
            uik=ui-uk;
            # Update the non linear model with new 'u'.
            MODEL.OverallSystem.updateSystem(t,MODEL.u,MODEL.speed)   
            # Recompute response force
            R = M.dot(a1*uik-va1) + C.dot(a4*uik+va2) + MODEL.OverallSystem.K_sys_upd.dot(MODEL.u) - MODEL.OverallSystem.f_int - fi;
            
            Rnorm = np.linalg.norm(R);
            
            nit = nit+1
        # Feedback
        if (nit>maxit):
            print('Step {} did not converge after {} iterations, residual {} \n'.format(i,nit, Rnorm/fnorm))
        # else:
        #     print('Step {} converged after {} iterations, residual {} \n'.format(i,nit, Rnorm/fnorm));
        #     # print('Rnorm',Rnorm,'tolfnorm',tol*fnorm)
        # Compute U,V,A
        uk = ui;
        vk = a4*uik+va2;
        ak = a1*uik-va1;
        # Update U,V,A
        MODEL.U[:,i+1]=uk[:,0];
        MODEL.V[:,i+1] = vk[:,0];
        MODEL.A[:,i+1] = ak[:,0];
        # iii+=1
        # if iii==100:
        #     u = np.linalg.solve(MODEL.OverallSystem.K_sys_upd,MODEL.OverallSystem.f_sys)
        #     x=np.append(np.hstack([np.array([0,0.22,0.38])+i*0.6 for i in range(MODEL.OverallSystem.n_sleepers)]),18.6)
        #     # x= np.reshape([xi*0.16,0.16+xi*0.22,0.38+xi*0.22],(-1))+0.6*0-0.5*n_sleepers_modes*0.6+0.3-0.08+0.3*loading
        #     plt.plot(x,u[[i for i,j in enumerate(MODEL.OverallSystem.u_names) if 'w_rail' in j]],'b')
        #     plt.plot(x,u[[i for i,j in enumerate(MODEL.OverallSystem.u_names) if 'w_rail' in j]],'r.')
        #     iii=0
    return MODEL
#%% Time Integration
class TimeIntegrationNM():
    def __init__(self,Overallsystem, u_0,speed = 18, t_end = 1,dt = 0.0001):
        """
        Initialize time integration parameters for Newmark solver

        Parameters
        ----------
        Overallsystem : Overall system assembly from RailFE.SystemAssembly
        u_0 : Initial conditions on DOF vector.
        speed : Speed integer. Currently varying speed no supported. The default is 18.
        t_end : Duration of simulation time in seconds. The default is 1.
        dt : Step size in seconds. The default is 0.0001.

        Returns
        -------
        None.

        """
        self.OverallSystem = Overallsystem
        self.speed = speed
        self.index = 0
        self.t_prev = 0
        self.t_start = 0.0
        self.t_end = t_end
        self.dt = dt
        self.nt = int((self.t_end-self.t_start)/self.dt)
        self.speed = speed
        self.u = u_0
    def shift(self,xx):
        """
        Shift xx on changing timoshenko element

        Parameters
        ----------
        xx : Initial Vector

        Returns
        -------
        xx : Shifted Vector

        """
        if self.OverallSystem.modal_analysis:
            eab = np.where(np.array(self.OverallSystem.u_names)=='w_axle_1')[0][0]+1 #end_of_body_assembly
            xx_shifted = np.zeros(xx.shape)
            xx_shifted[:eab] = xx[:eab]
            shift = len(self.OverallSystem.repeated_segments.keys())*2+1
            shift1 = self.OverallSystem.Track.Timoshenko4.modal_n_modes*len(self.OverallSystem.repeated_segments.keys())
            xx_shifted[eab+shift:-shift-len(self.OverallSystem.Local.u_names_l)] = xx[eab+2*shift:-len(self.OverallSystem.Local.u_names_l)]
            xx_shifted[-len(self.OverallSystem.Local.u_names_l)+shift1:-shift1] = xx[-len(self.OverallSystem.Local.u_names_l)+2*shift1:]
            xx = xx_shifted         
        else:
            eab = np.where(np.array(self.OverallSystem.u_names)=='w_axle_1')[0][0]+1 #end_of_body_assembly
            xx_shifted = np.zeros(xx.shape)
            xx_shifted[:eab] = xx[:eab]
            shift = len(self.OverallSystem.repeated_segments.keys())*2+1
            xx_shifted[eab+shift:-shift] = xx[eab+2*shift:]
            xx = xx_shifted
        return xx
    def newmark_slv(self):
        """
        Solve the time integration with Newmark Solver

        Returns
        -------
        bool: return True if successfull, else return False

        """
        try:
            newmark_nl(self)
            return True
        except Exception as e:
            print('Exception: {}'.format(e))
            return False