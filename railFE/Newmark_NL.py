# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:53:43 2020

@author: CyprienHoelzl
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
#%% Trash
            # kc = MODEL.OverallSystem.Track.Timoshenko4.railproperties.K_c
            # delta_u_rw = np.dot(E[:-1,0],Yi[idx0])
            # if delta_u_rw>=0:
            #     K_Hi0 = np.sort(np.roots((alpha,1,0,-kc**2*delta_u_rw)))[-1]
            #     # where delta_u_rw = u_w - N_i*u_i
            #     print(K_Hi0)
            # else:
            #     K_Hi0 = 0
            # K_H = K_Hi0/(1+ alpha*K_Hi0) # Linearized contact stiffness
            #P_i0 = K_H * MODEL.OverallSystem.Local.E[0,:]
            #print('select root')
            #MODEL.Kci = MODEL.OverallSystem.Local.E*K_H
# def assemble_nl( MODEL ):
    
#     ndof = 6*numel(MODEL.nodes[:,0]);
    
#     nt = MODEL.nt;
#     MODEL.K = scipy.sparse.csc_matrix(ndof,ndof);
#     MODEL.M = sparse(ndof,ndof);
#     MODEL.fint = np.zeros((ndof,1));
#     MODEL.f = np.zeros((ndof,1));
    
#     if (isfield(MODEL,'u')==0)
#         MODEL.u = np.zeros((ndof,1));
#     end
    
#     % Assembly of the beam elements
    
#     n_beams =size(MODEL.beam_elements,1);
    
#     for e=1:n_beams
#         beam_type = MODEL.beam_elements(e,1);
#         beam_nodes = MODEL.beam_elements(e,2:end);
#         beam_COORDS = MODEL.nodes(beam_nodes,:);
#         beam_material_properties = MODEL.material_properties(MODEL.beam_material_properties(e),:);
#         beam_cross_section_properties = MODEL.cross_sections(MODEL.beam_cross_sections(e,:),:);
#         beam_loads = MODEL.beam_loads(e,:);
        
#         [ beam_dofs ] = element_global_dofs( beam_nodes );
#         [Ke,Me,fe] = beam_mass_stiffness_rhs(beam_type, beam_COORDS, beam_material_properties, beam_cross_section_properties, beam_loads);
#         ue = MODEL.u(beam_dofs);
    
#         MODEL.K(beam_dofs,beam_dofs) = MODEL.K(beam_dofs,beam_dofs) + Ke;
#         MODEL.M(beam_dofs,beam_dofs) = MODEL.M(beam_dofs,beam_dofs) + Me;
#         MODEL.fint(beam_dofs) = MODEL.fint(beam_dofs) + Ke*ue;
#         MODEL.f(beam_dofs) = MODEL.f(beam_dofs) + fe;
#     end
    
#     # % Assembly of the plate elements
    
#     # if (isfield(MODEL,'plate_elements'))
    
#     #     n_plates = numel(MODEL.plate_elements[:,0]);
#     # else
#     #     n_plates = 0;
#     # end
    
#     # for e=1:n_plates
#     #     plate_nodes = MODEL.plate_elements(e,1:end);
#     #     plate_COORDS = MODEL.nodes(plate_nodes,:);
#     #     plate_material_properties = MODEL.material_properties(MODEL.plate_material_properties(e),:);
#     #     plate_thickness = MODEL.plate_thickness(e,:);
#     #     plate_loads = MODEL.plate_loads(e,:);
        
#     #     [Ke,Me,fe] = shell_mass_stiffness_rhs( plate_COORDS, plate_material_properties, plate_thickness, plate_loads);
#     #     [ plate_dofs ] = element_global_dofs( plate_nodes );
#     #     ue = MODEL.u(plate_dofs);
    
#     #     MODEL.K(plate_dofs,plate_dofs) = MODEL.K(plate_dofs,plate_dofs) + Ke;
#     #     MODEL.M(plate_dofs,plate_dofs) = MODEL.M(plate_dofs,plate_dofs) + Me;
#     #     MODEL.fint(plate_dofs) = MODEL.fint(plate_dofs) + Ke*ue;
#     #     MODEL.f(plate_dofs) = MODEL.f(plate_dofs) + fe;
#     # end
    
#     % Assembly of nonlinear links
    
#     if (isfield(MODEL,'nl_link_elements'))
#         n_links = numel(MODEL.nl_link_elements[:,0]);
#         ks = max(max(diag(MODEL.K)))*1e3;
    
#         if (~isfield(MODEL,'nl_link_hist'))
#             MODEL.nl_link_hist = cell(n_links,1);
#             MODEL.nl_link_hist(:) ={struct('uj',{0,0,0,0,0,0},'rj',{0,0,0,0,0,0},'kj',{0,0,0,0,0,0})};
#         end
#     else
#         n_links = 0;
#     end
    
#     for e=1:n_links
        
#         link_nodes = MODEL.nl_link_elements(e,1:end);
#         link_properties = MODEL.nl_link_bw_properties(MODEL.nl_link_properties(e,:),:);
#         link_flags = MODEL.nl_link_flags(e,:);
#         link_hist = MODEL.nl_link_hist{e,:};
#         [ link_dofs ] = element_global_dofs( link_nodes );
        
#         ue = MODEL.u(link_dofs);
        
#         [finte, Ke, link_hist] = link_residual_stiffness(ue, link_properties, link_flags, link_hist, ks);
        
#         MODEL.nl_link_hist{e,:} = link_hist;
        
#         if e==1
#             if (isfield(MODEL,'Hist')==1)
#                 MODEL.Hist(1,nt) =link_hist(5).uj;
#                 MODEL.Hist(2,nt) = link_hist(5).rj;
#             end
#         end
        
#         if e==9
#             if (isfield(MODEL,'Hist')==1)
#                 MODEL.Hist(3,nt) =link_hist(5).uj;
#                 MODEL.Hist(4,nt) = link_hist(5).rj;
#             end
#         end
    
#         MODEL.K(link_dofs,link_dofs) = MODEL.K(link_dofs,link_dofs) + Ke;
#         MODEL.fint(link_dofs) = MODEL.fint(link_dofs) + finte;
#     end
    
#     % Assembly of the springs
    
#     if (isfield(MODEL,'springs')&&(numel(MODEL.springs)>0))
    
#         n_springs = numel(MODEL.springs[:,0]);
#     else
#         n_springs = 0;
#     end
    
#     for s=1:n_springs
#         node = MODEL.springs(s,1);
#         spring_constants = MODEL.springs(s,2:end);
#         [ dofs ] = node_global_dofs( node );
#         MODEL.K(dofs,dofs) = MODEL.K(dofs,dofs) + diag(eye(numel(dofs))*spring_constants');
#     end
    
#     % Assembly of masses
    
#     if (isfield(MODEL,'masses')&&(numel(MODEL.masses)>0))
    
#         n_masses = numel(MODEL.masses[:,0]);
#     else
#         n_masses = 0;
#     end
    
#     for m=1:n_masses
#         node = MODEL.masses(m,1);
#         mass = MODEL.masses(m,2:end);
#         [ dofs ] = node_global_dofs( node );
#         MODEL.M(dofs,dofs) = MODEL.M(dofs,dofs) + diag(eye(numel(dofs))*mass');
#     end
# def apply_bc_nl( MODEL )

#     ndofs = numel(MODEL.nodes[:,0])*6;
    
#     % Nodal loads
    
#     if (numel(MODEL.nodal_loads)>0)
    
#         n_loads = numel(MODEL.nodal_loads[:,0]);
    
#         for l=1:n_loads
#             node = MODEL.nodal_loads(l,1);
#             load = MODEL.nodal_loads(l,2:end);
#             [ dofs ] = node_global_dofs( node );
#             MODEL.f(dofs) = MODEL.f(dofs) + load';
#         end
    
#     end
    
#     % Imposed displacements
    
#     if (numel(MODEL.nodal_displacements)>0)
        
#         bc_dofs = [];
#         bc_dofs_nz = [];
#         bc_values = [];
    
#         for d=1:numel(MODEL.nodal_displacements[:,0])
#             node = MODEL.nodal_displacements(d,1);
#             [ dofs ] = node_global_dofs( node );
#             dofs_act = dofs(find(MODEL.nodal_displacements(d,2:7)));
#             bc_dofs    = [bc_dofs; dofs_act'];
#             nz = find(MODEL.nodal_displacements(d,8:end));
#             if (numel(nz)>0)
#                 bc_dofs_nz = [bc_dofs_nz; dofs(nz)'];
#                 bc_values = [bc_values; MODEL.nodal_displacements(d,nz+7)'];
#             end
#         end
        
#         if (numel(bc_dofs_nz)>0)
#             fbc = MODEL.K(:,bc_dofs_nz)*bc_values;
#             MODEL.f = MODEL.f - fbc;
#         end
    
#         MODEL.K(bc_dofs, :) = 0;
#         MODEL.K(:, bc_dofs) = 0;
#         MODEL.M(bc_dofs, :) = 0;
#         MODEL.M(:, bc_dofs) = 0;
#         MODEL.f(bc_dofs) = 0;
#         MODEL.fint(bc_dofs) = 0;
        
#         if (numel(bc_dofs_nz)>0)
#             MODEL.f(bc_dofs_nz)=bc_values;
#             MODEL.fint(bc_dofs_nz) = 0;
#         end
            
#         diag_el = (bc_dofs-1)*(ndofs + 1) + 1;
    
#         MODEL.K(diag_el)=1;
#         MODEL.M(diag_el)=1;
#     end
