# -*- coding: utf-8 -*-
"""Rail beam formulation
  
For the rail formulation a timoshenko 4, 4DOF element is used 
with nodal displacement vector u^{(e)} = {w_1,\theta_1,w_2,\theta_2}

Extentions:
    - A TIM7 FE element could also be used, and  would fullfill C(1).
    - Currently this package is insufficiently documented

Usage:
    See examples

@author: 
    CyprienHoelzl
"""
import numpy as np
import scipy.integrate as integrate
import scipy
from collections.abc import Iterable   # drop `.abc` with Python 2.7 or lower
def IsIterable(obj):
    return isinstance(obj, Iterable)

class Timoshenko4():
    """
    Implementation of the Timoshenko4 beam using the formulation in:
        https://www.sciencedirect.com/science/article/pii/0045794993902437
        https://link.springer.com/content/pdf/10.1007/s00466-009-0431-2.pdf
        http://people.duke.edu/~hpgavin/cee541/StructuralElements.pdf
    
    Static analysis of Timoshenko beam resting on elastic half-plane
    based on the coupling of locking-free finite elements and boundary
    integral proposed by Nerio Tullini · Antonio Tralli
    """
    def __init__(self,railproperties,elementlength,modal_analysis  =True,modal_n_modes=10):
        self.railproperties = railproperties
        self.L = elementlength
        self.modal_analysis=modal_analysis
        self.Psi = railproperties.Psi(elementlength)
        
        if self.modal_analysis==True:
            self.modal_n_modes = modal_n_modes
            self.modal_nat_freqs = self.GetNaturalFrequencies(self.modal_n_modes)
            self.modal_P_coeffs  = [self.CalcP(i) for i in self.modal_nat_freqs]
    def N_w1(self,xi):
        return 1/(1+self.Psi)*(1-3*xi**2+2*xi**3+self.Psi*(1-xi))
    def N_t1(self,xi):
        return self.L*xi/(1+self.Psi)*((1-xi)**2+self.Psi/2*(1-xi))
    def N_w2(self,xi):
        return 1/(1+self.Psi)*(-2*xi**3+3*xi**2+self.Psi*xi)
    def N_t2(self,xi):
        return self.L*xi/(1+self.Psi)*((xi**2-xi)-self.Psi/2*(1-xi))
    
    def Nt_w1(self,xi):
        return 1/(1+self.Psi)*(6*xi*(1-xi))/self.L
    def Nt_t1(self,xi):
        return 1/(1+self.Psi)*(1-4*xi+3*xi**2+self.Psi*(1-xi))
    def Nt_w2(self,xi):
        return -1/(1+self.Psi)*(6*xi*(1-xi))/self.L
    def Nt_t2(self,xi):
        return 1/(1+self.Psi)*(-2*xi+3*xi**2+self.Psi*xi)
    # def N_e(self,xi):
    #     N_e_tim4 = [self.N_w1(xi), 
    #                 self.N_t1(xi), 
    #                 self.N_w2(xi), 
    #                 self.N_t2(xi)]
    #     return N_e_tim4
    def stiffnessMatrix(self):
        i1,i2 = np.triu_indices(4)
        K_ef = np.zeros((4,4))
        K_ef[0,0] = 12/self.L**2
        K_ef[1,1] = (4+self.Psi)
        K_ef[2,2] = 12/self.L**2
        K_ef[3,3] = (4+self.Psi)
        
        K_ef[0,1] = (6/self.L)
        K_ef[0,2] = -(12/self.L**2)
        K_ef[0,3] = (6/self.L)
        K_ef[1,0] = K_ef[0,1]
        K_ef[2,0] = K_ef[0,2]
        K_ef[3,0] = K_ef[0,3]
        K_ef[1,2] = -(6/self.L)
        K_ef[2,1] = K_ef[1,2]
        K_ef[1,3] = (2-self.Psi)
        K_ef[3,1] = K_ef[1,3]
        K_ef[2,3] = -(6/self.L)
        K_ef[3,2] = K_ef[2,3]
        
        K_ef = 1/(1+self.Psi)*self.railproperties.E*self.railproperties.I/self.L*K_ef
        self.K_EF = K_ef
        return K_ef
    def massMatrix(self):
        #i1,i2 = np.triu_indices(4)
        M_ef = np.zeros((4,4))
        M_ef[0,0] = (78+147*self.Psi+70*self.Psi**2)
        M_ef[1,1] = (8 + 14*self.Psi+7*self.Psi**2)*self.L**2/4
        M_ef[2,2] = (78+147*self.Psi+70*self.Psi**2)
        M_ef[3,3] = (8 + 14*self.Psi+7*self.Psi**2)*self.L**2/4
        
        M_ef[0,1] = (44 + 77*self.Psi+35*self.Psi**2)*self.L/4
        M_ef[0,2] = (27 + 63*self.Psi+35*self.Psi**2)
        M_ef[0,3] = -(26 + 63*self.Psi+35*self.Psi**2)*self.L/4
        M_ef[1,0] = M_ef[0,1]
        M_ef[2,0] = M_ef[0,2]
        M_ef[3,0] = M_ef[0,3]
        M_ef[1,2] = (26 + 63*self.Psi+35*self.Psi**2)*self.L/4
        M_ef[2,1] = M_ef[1,2]
        M_ef[1,3] = -(6 + 14*self.Psi+7*self.Psi**2)*self.L**2/4
        M_ef[3,1] = M_ef[1,3]
        M_ef[2,3] = -(44 + 77*self.Psi+35*self.Psi**2)*self.L/4
        M_ef[3,2] = M_ef[2,3]
        
        M_efI = np.zeros((4,4))
        M_efI[0,0] = 36
        M_efI[1,1] = (4 + 5*self.Psi+10*self.Psi**2)*self.L**2
        M_efI[2,2] = 36
        M_efI[3,3] = (4 + 5*self.Psi+10*self.Psi**2)*self.L**2
        
        M_efI[0,1] = -(-3 + 15*self.Psi)*self.L
        M_efI[0,2] = -36
        M_efI[0,3] = -(-3 + 15*self.Psi)*self.L
        M_efI[1,0] = M_efI[0,1]
        M_efI[2,0] = M_efI[0,2]
        M_efI[3,0] = M_efI[0,3]
        M_efI[1,2] = (-3 + 15*self.Psi)*self.L
        M_efI[2,1] = M_efI[1,2]
        M_efI[1,3] = (-1 - 5*self.Psi+5*self.Psi**2)*self.L**2
        M_efI[3,1] = M_efI[1,3]
        M_efI[2,3] = (-3 + 15*self.Psi)*self.L
        M_efI[3,2] = M_efI[2,3]
        
        M_ef = ((self.railproperties.rho*self.railproperties.A*self.L)/(210*(1+self.Psi)**2)*M_ef +
                ((self.railproperties.rho*self.railproperties.I)/((30*(1+self.Psi)**2)*self.L))*M_efI)
        self.M_EF = M_ef
        return M_ef       
    def dampingMatrix(self):
        #i1,i2 = np.triu_indices(4)
        self.D_EF = 0*(1/100*self.M_EF+1/1000*self.K_EF)
        return self.D_EF
    def point_load_flexibility_TIM4(self,xi):
        # A unit load on the TIM4 at point xi[0,1] causes a displacement of F_loc at point xi.
        # F_loc [m/kN]
        L   = self.L
        Psi = self.Psi
        E   = self.railproperties.E
        I   = self.railproperties.I
        T_loc = -(L**3/(6*E*I)*((xi*((1-xi)*(xi**2-Psi/2)+3*xi**2-xi**3-2*xi))+
                              (xi*(1-xi)/(2*(1+Psi))*((2-2*xi+Psi)*(xi**3-3*xi**2+2*xi)+(2*xi+Psi)*(xi-xi**3)))))
        return T_loc  
    
    
    def Lambda(self,omega):
        rho = self.railproperties.rho
        k   = self.railproperties.k
        A   = self.railproperties.A
        G   = self.railproperties.G
        I   = self.railproperties.I
        E   = self.railproperties.E
        L   = self.L
        omega_0 = np.sqrt(k*A*G/(rho*I)) # cut off frequency
        #Wavenumbers http://yadda.icm.edu.pl/yadda/element/bwmeta1.element.baztech-article-BWM4-0019-0012/c/httpwww_ptmts_org_plmajkut-1-09.pdf Majkut L. Free and forced vibrations of Timoshenko beams described by single difference equation. J Theor Appl Mech. 2009;47(1):193–210.
     
        p = omega**2*rho/(k*G)+k*A*G/(E*I)
        d = omega**2*rho*I*(1+E/(G*k))/(E*I)
        e = omega**2*(omega**2*rho**2*I/(G*k)-rho*A)/(E*I)
        delta = d**2-4*e
      
        if omega<omega_0:               
            lambda_1 = np.sqrt((d + np.sqrt(delta))/2)
            lambda_2 = np.sqrt((-d + np.sqrt(delta))/2)
            Lambda = np.array([[1,0,1,0],
                         [0,lambda_1*(p-lambda_1**2),0,lambda_2*(p+lambda_2**2)],
                         [np.cos(lambda_1*L),np.sin(lambda_1*L),np.cosh(lambda_2*L),np.sinh(lambda_2*L)],
                         [np.sin(lambda_1*L)*(lambda_1**3-p*lambda_1),np.cos(lambda_1*L)*(-lambda_1**3+p*lambda_1),
                          np.sinh(lambda_2*L)*(lambda_2**3+p*lambda_2),np.cosh(lambda_2*L)*(lambda_2**3+p*lambda_2)]])
        elif omega > omega_0:
            lambda_1 = np.sqrt((d + np.sqrt(delta))/2)
            lambda_2 = np.sqrt((d - np.sqrt(delta))/2)
            Lambda = np.array([[1,0,1,0],
                              [0,lambda_1*(p-lambda_1**2),0,lambda_2*(p-lambda_2**2)],
                              [np.cos(lambda_1*L),np.sin(lambda_1*L),np.cos(lambda_2*L),np.sin(lambda_2*L)],
                              [np.sin(lambda_1*L)*(lambda_1**3-p*lambda_1),np.cos(lambda_1*L)*(-lambda_1**3+p*lambda_1),
                               np.sin(lambda_2*L)*(lambda_2**3-p*lambda_2),np.cos(lambda_2*L)*(-lambda_2**3+p*lambda_2)]])
        else:
            raise Warning('no compatible omega, omega = {}, omega_0 = {}'.format(omega,omega_0))
             
        return Lambda
    def DetLambda(self,omega):
        return np.linalg.det(self.Lambda(omega))
    def CalcP(self,omega):
        # Compute the coefficients of P
        Lambda_o = self.Lambda(omega)
        # np.det(Lamda)
        firstcol = Lambda_o[:,0].copy()
        # we impose a condition that first term be 1, 
        x = np.linalg.lstsq(Lambda_o[:, 1:], -firstcol,rcond = None)[0]
        x = np.r_[1, x]
        # Normalize them by integral(rho*A*W**2+rho*I*Theta**2,0,L)
        P = self.NormalizeP(x,omega)
        return P        
    def NormalizeP(self,P,omega):
        rho = self.railproperties.rho
        A   = self.railproperties.A
        I   = self.railproperties.I
        norm = self.L*integrate.quad(lambda x: rho*A*self.W(x,P,omega)**2 +rho*I*self.T(x,P,omega)**2,0,1)[0]
        
        # scipy.optimize.brentq(lambda x: integrate.quad(lambda x: rho*A*self.W(x,P,omega)**2 +rho*I*self.T(x,P,omega)**2,0,1)[0]-1, 
        
        P /= norm**0.5
        return P
    def preprocess(self,f,xmin,xmax,step,args=()):  
        if not isinstance(args, tuple):
                args = (args,)
        # Find when the function changes sign. Subdivide
        first_sign = f(xmin) > 0 # True if f(xmin) > 0, otherwise False
        x = xmin + step
        while x <= xmax: # This loop detects when the function changes its sign
            fstep = f(x,*args)
            if first_sign and fstep < 0:
                return x
            elif not(first_sign) and fstep > 0:
                return x
            x += step
        return x # If you ever reach here, that means that there isn't a zero in the function    
    def subdividespace(self,f,xmin,xmax,step,args=()):   
        if not isinstance(args, tuple):
                args = (args,)
        x_list = []
        x = xmin
        while x < xmax: # This discovers when f changes its sign
            x_list.append(x)
            x = self.preprocess(f,x,xmax,step,args)
        # x_list.append(xmax)
        return x_list
    def GetNaturalFrequencies(self,n=5):
        # Gets the first n natural vibration frequencies of timoshenko beam.
        rho = self.railproperties.rho
        k   = self.railproperties.k
        A   = self.railproperties.A
        G   = self.railproperties.G
        I   = self.railproperties.I
        omega_0 = np.sqrt(k*A*G/(rho*I)) # cut off frequency
        # Frequencies in Rad/s
        freq_0 = 10    # Hz 
        freq_1 = 60000 # Hz
        step   = 20   # Hz
        # Get the n first natural frequencies of beam vibration
        x_list = self.subdividespace(self.DetLambda,freq_0*np.pi*2,freq_1*np.pi*2,step*np.pi*2)
        z_list = []
        for i in range(len(x_list) - 1):
            nat_freq = scipy.optimize.brentq(self.DetLambda,x_list[i],x_list[i + 1])
            if abs(nat_freq-omega_0)>50:
                z_list.append(nat_freq)
        return sorted(z_list)[:n]
    def W(self,xi,P,omega):
        P1,P2,P3,P4 = P
        rho = self.railproperties.rho
        k   = self.railproperties.k
        A   = self.railproperties.A
        G   = self.railproperties.G
        I   = self.railproperties.I
        E   = self.railproperties.E
        L   = self.L
        omega_0 = np.sqrt(k*A*G/(rho*I)) # cut off frequency
       
        # p = omega**2*rho/(k*G)+k*A*G/(E*I)
        d = omega**2*rho*I*(1+E/(G*k))/(E*I)
        e = omega**2*(omega**2*rho**2*I/(G*k)-rho*A)/(E*I)
        delta = d**2-4*e
        x = xi*L
       
        if omega<omega_0:  
            lambda_1 = np.sqrt((d + np.sqrt(delta))/2)
            lambda_2 = np.sqrt((-d + np.sqrt(delta))/2) 
            W = P1*np.cos(lambda_1*x) + P2*np.sin(lambda_1*x) + P3*np.cosh(lambda_2*x) + P4*np.sinh(lambda_2*x )
        elif omega > omega_0:
            lambda_1 = np.sqrt((d + np.sqrt(delta))/2)
            lambda_2 = np.sqrt((d - np.sqrt(delta))/2)
            W = P1*np.cos(lambda_1*x) + P2*np.sin(lambda_1*x) + P3*np.cos(lambda_2*x) + P4*np.sin(lambda_2*x )
        return W
    def T(self,xi,P,omega):
        rho = self.railproperties.rho
        k   = self.railproperties.k
        A   = self.railproperties.A
        G   = self.railproperties.G
        I   = self.railproperties.I
        E   = self.railproperties.E
        L   = self.L
        omega_0 = np.sqrt(k*A*G/(rho*I)) # cut off frequency

        x = xi*L
        if omega<omega_0:  
            P1,P2,P3,P4 = P
            T = (((rho*omega**2)/(G*k) + (A*G*k)/(E*I))*(P2*np.cos(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2) + P4*np.cosh(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2) - P1*np.sin(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2) + P3*np.sinh(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2)) - P2*np.cos(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(3/2) + P4*np.cosh(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(3/2) + P1*np.sin(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(3/2) + P3*np.sinh(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(3/2))/((k*A*G)/(E*I) - (omega**2*rho)/E)
        elif omega > omega_0:
            Q1,Q2,Q3,Q4 = P
            T = (((rho*omega**2)/(G*k) + (A*G*k)/(E*I))*(Q2*np.cos(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2) + Q4*np.cos(x*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(1/2))*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(1/2) - Q1*np.sin(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2) - Q3*np.sin(x*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(1/2))*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(1/2)) - Q2*np.cos(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(3/2) - Q4*np.cos(x*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(1/2))*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(3/2) + Q1*np.sin(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(3/2) + Q3*np.sin(x*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(1/2))*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(3/2))/((k*A*G)/E*I - (omega**2*rho)/E)   
        return T
    
    
    def F_res(self,xi):
        omega_l = self.modal_nat_freqs
        F_Mod = 0
        for i in range(self.modal_n_modes):
            omega = omega_l[i]
            P = self.modal_P_coeffs[i]
            F_Mod = F_Mod + self.W(xi,P,omega)**2/omega**2
        F_res = self.point_load_flexibility_TIM4(xi) - F_Mod   
        return F_res
    def M_cross_dyn(self):
        # M_ij^cross where i is i-th mode of vibration and j is j-th global dof
        rho = self.railproperties.rho
        A   = self.railproperties.A
        I   = self.railproperties.I
        
        omega_l = self.modal_nat_freqs
        M_i_cross = np.zeros((self.modal_n_modes,4))
        for i in range(self.modal_n_modes):
            omega = omega_l[i]
            P = self.modal_P_coeffs[i] 
            M_i1 = self.L*integrate.quad(lambda x: rho*A*self.W(x,P,omega)*self.N_w1(x) +rho*I*self.T(x,P,omega)*self.Nt_w1(x) ,0,1)[0]
            M_i2 = self.L*integrate.quad(lambda x: rho*A*self.W(x,P,omega)*self.N_t1(x) +rho*I*self.T(x,P,omega)*self.Nt_t1(x) ,0,1)[0]
            M_i3 = self.L*integrate.quad(lambda x: rho*A*self.W(x,P,omega)*self.N_w2(x) +rho*I*self.T(x,P,omega)*self.Nt_w2(x) ,0,1)[0]
            M_i4 = self.L*integrate.quad(lambda x: rho*A*self.W(x,P,omega)*self.N_t2(x) +rho*I*self.T(x,P,omega)*self.Nt_t2(x) ,0,1)[0]
            M_i_cross[i,:] = [M_i1,M_i2,M_i3,M_i4]
        return M_i_cross
    def M_loc_dyn(self):
        return np.eye(self.modal_n_modes)
    def K_loc_dyn(self):
        omega_l = self.modal_nat_freqs
        return np.diag(np.array(omega_l)**2)
    def C_loc_dyn(self):
        omega_l = np.array(self.modal_nat_freqs)
        alpha = 5000
        beta  = 0.0000004
        zeta = alpha/(2*omega_l)+beta*omega_l/2
        return np.diag(2*np.multiply(omega_l,zeta))
    def Psi_Modal(self,xi):
        Psi_modal =[]
        for i in range(self.modal_n_modes):
            omega = self.modal_nat_freqs[i]
            P     = self.modal_P_coeffs[i]
            Psi_modal.append(self.W(xi,P,omega))
        return Psi_modal
# class Timoshenko6():
#     # https://mycourses.aalto.fi/pluginfile.php/206112/mod_resource/content/1/NMSE-16-Lectures7.pdf
#     def __init__(self,railproperties,elementlength):
#         self.railproperties = railproperties
#         self.L = elementlength
#         self.Psi = railproperties.Psi(elementlength)
#     def N_w1(self,xi):
#         return 1/(1+self.Psi)*(1-3*xi**2+2*xi**3+self.Psi*(1-xi))
#     def N_t1(self,xi):
#         return self.L*xi/(1+self.Psi)*((1-xi)**2+self.Psi/2*(1-xi))
#     def N_w2(self,xi):
#         return 1/(1+self.Psi)*(-2*xi**3+3*xi**2+self.Psi*xi)
#     def N_t2(self,xi):
#         return self.L*xi/(1+self.Psi)*((xi**2-xi)-self.Psi/2*(1-xi))       
class Timoshenko4eb():
    """
    # Implementation of the Timoshenko4 beam on an elastic bedding.
      Static analysis of Timoshenko beam resting on elastic half-plane
      based on the coupling of locking-free finite elements and boundary
      integral
    """
    def  __init__(self,railproperties,padproperties,elementlength,modal_analysis = True,modal_n_modes=10):
        self.railproperties = railproperties
        padproperties.distributed_params(elementlength)
        self.padproperties = padproperties
        self.L = elementlength
        self.modal_analysis=modal_analysis
        self.Psi = railproperties.Psi(elementlength)
        self.calc_alpha_beta()
        self.coeffs = self.interpfunction_coeffs()
        
        if self.modal_analysis==True:
            self.modal_n_modes   = modal_n_modes
            self.modal_nat_freqs = self.GetNaturalFrequencies(self.modal_n_modes)
            self.modal_P_coeffs  = [self.CalcP(i) for i in self.modal_nat_freqs]
    def calc_alpha_beta(self):
        # Timoshenko:
        #   w'''' + 0* w''' - k_pd/kAG*w'' + 0*w' + k_pd/EI*w = 0
        #  r^4 - C1*r^2 + C2 * r = r*(r^3-C1*r+C2) = 
        C1 = self.padproperties.k_pd/(self.railproperties.k*self.railproperties.A*self.railproperties.G)
        C2 = self.padproperties.k_pd/(self.railproperties.E*self.railproperties.I)
        roots_diffeq0 = np.roots([1,0,-C1,0,C2])
        roots_diffeq = roots_diffeq0[np.iscomplex( roots_diffeq0)]
        beta = np.abs(np.real(roots_diffeq[0]))
        alpha = np.abs(np.imag(roots_diffeq[0]))
        self.beta = beta
        self.alpha = alpha
        # print(roots_diffeq[0]**4-C1*roots_diffeq[0]**2+C2*roots_diffeq[0])
        # print('Characteristic roots: ' + ', '.join([str(i) for i in roots_diffeq0]))
        # print('Alpha: {}, Beta: {}'.format(str(alpha),str(beta)))
        return (alpha,beta)
    def interpfunction_coeffs(self):
        L = self.L
        alpha,beta = self.alpha,self.beta
        # Solving the Timoshenko Beam with elastic bedding, we obtain:
        #   w(x)     = C_1 g_1 (x) + C_2 g_2 (x) + C_3 g_3 (x) + C_4 g_4 (x)
        g1_0 = 1 
        g2_0 = 0 
        g3_0 = 0 
        g4_0 = 0
        g1_L = np.cosh(beta*L)*np.cos(alpha*L)
        g2_L = np.cosh(beta*L)*np.sin(alpha*L)
        g3_L = np.sinh(beta*L)*np.cos(alpha*L)
        g4_L = np.sinh(beta*L)*np.sin(alpha*L)
        # Substituting in theta:
        #   theta = w' * (1- k_d*EI/(kAG)^2) +EI/(kAG)*w'''
        #   theta(x) = C_1 h_1 (x) + C_2 h_2 (x) + C_3 h_3 (x) + C_4 h_4 (x)
        # C1_coeffs(x) = EIkAG*(alpha^3*cosh(beta*x)*sin(alpha*x) + beta^3*cos(alpha*x)*sinh(beta*x) - 3*alpha*beta^2*cosh(beta*x)*sin(alpha*x) - 3*alpha^2*beta*cos(alpha*x)*sinh(beta*x)) + (alpha*cosh(beta*x)*sin(alpha*x) - beta*cos(alpha*x)*sinh(beta*x))*(kdEIkAg2 - 1)
        # C2_coeffs(x) = - EIkAG*(alpha^3*cos(alpha*x)*cosh(beta*x) - beta^3*sin(alpha*x)*sinh(beta*x) - 3*alpha*beta^2*cos(alpha*x)*cosh(beta*x) + 3*alpha^2*beta*sin(alpha*x)*sinh(beta*x)) - (beta*sin(alpha*x)*sinh(beta*x) + alpha*cos(alpha*x)*cosh(beta*x))*(kdEIkAg2 - 1)
        # C3_coeffs(x) = EIkAG*(beta^3*cos(alpha*x)*cosh(beta*x) + alpha^3*sin(alpha*x)*sinh(beta*x) - 3*alpha^2*beta*cos(alpha*x)*cosh(beta*x) - 3*alpha*beta^2*sin(alpha*x)*sinh(beta*x)) + (alpha*sin(alpha*x)*sinh(beta*x) - beta*cos(alpha*x)*cosh(beta*x))*(kdEIkAg2 - 1)
        # C4_coeffs(x) = - EIkAG*(alpha^3*cos(alpha*x)*sinh(beta*x) - beta^3*cosh(beta*x)*sin(alpha*x) - 3*alpha*beta^2*cos(alpha*x)*sinh(beta*x) + 3*alpha^2*beta*cosh(beta*x)*sin(alpha*x)) - (alpha*cos(alpha*x)*sinh(beta*x) + beta*cosh(beta*x)*sin(alpha*x))*(kdEIkAg2 - 1)  
        EIkAG =self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)
        kdEIkAg2  = self.padproperties.k_pd*self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)**2
        h1_0 = 0 
        h2_0 = EIkAG*(3*alpha*beta**2 - alpha**3) - alpha*(kdEIkAg2 - 1) 
        h3_0 = - beta*(kdEIkAg2 - 1) - EIkAG*(3*alpha**2*beta - beta**3)
        h4_0 = 0
        h1_L = EIkAG*(alpha**3*np.cosh(L*beta)*np.sin(L*alpha) + beta**3*np.cos(L*alpha)*np.sinh(L*beta) - 3*alpha*beta**2*np.cosh(L*beta)*np.sin(L*alpha) - 3*alpha**2*beta*np.cos(L*alpha)*np.sinh(L*beta)) + (alpha*np.cosh(L*beta)*np.sin(L*alpha) - beta*np.cos(L*alpha)*np.sinh(L*beta))*(kdEIkAg2 - 1)
        h2_L = - EIkAG*(alpha**3*np.cos(L*alpha)*np.cosh(L*beta) - beta**3*np.sin(L*alpha)*np.sinh(L*beta) - 3*alpha*beta**2*np.cos(L*alpha)*np.cosh(L*beta) + 3*alpha**2*beta*np.sin(L*alpha)*np.sinh(L*beta)) - (alpha*np.cos(L*alpha)*np.cosh(L*beta) + beta*np.sin(L*alpha)*np.sinh(L*beta))*(kdEIkAg2 - 1)
        h3_L = EIkAG*(beta**3*np.cos(L*alpha)*np.cosh(L*beta) + alpha**3*np.sin(L*alpha)*np.sinh(L*beta) - 3*alpha**2*beta*np.cos(L*alpha)*np.cosh(L*beta) - 3*alpha*beta**2*np.sin(L*alpha)*np.sinh(L*beta)) - (beta*np.cos(L*alpha)*np.cosh(L*beta) - alpha*np.sin(L*alpha)*np.sinh(L*beta))*(kdEIkAg2 - 1)
        h4_L = - EIkAG*(alpha**3*np.cos(L*alpha)*np.sinh(L*beta) - beta**3*np.cosh(L*beta)*np.sin(L*alpha) - 3*alpha*beta**2*np.cos(L*alpha)*np.sinh(L*beta) + 3*alpha**2*beta*np.cosh(L*beta)*np.sin(L*alpha)) - (alpha*np.cos(L*alpha)*np.sinh(L*beta) + beta*np.cosh(L*beta)*np.sin(L*alpha))*(kdEIkAg2 - 1)
            
        GH = np.array([[g1_0, g2_0, g3_0, g4_0],
                      [h1_0, h2_0, h3_0, h4_0],
                      [g1_L, g2_L, g3_L, g4_L],
                      [h1_L, h2_L, h3_L, h4_L]])

        # N_w1_EF when w(0)=1, theta(0)=0, w(L)=0, theta(L)=0
        N_w1_EF = np.linalg.solve(GH, [1,0,0,0])
        # N_t1_EF when w(0)=0, theta(0)=1, w(L)=0, theta(L)=0
        N_t1_EF = np.linalg.solve(GH, [0,1,0,0])
        # N_w2_EF when w(0)=0, theta(0)=0, w(L)=1, theta(L)=0
        N_w2_EF = np.linalg.solve(GH, [0,0,1,0])
        # N_t2_EF when w(0)=0, theta(0)=0, w(L)=0, theta(L)=1
        N_t2_EF = np.linalg.solve(GH, [0,0,0,1])
        
        # self.coeffs = (N_w1_EF,N_t1_EF,N_w2_EF,N_t2_EF)
        N_s = self.particular_solution_sleeper()
        return (N_w1_EF,N_t1_EF,N_w2_EF,N_t2_EF,N_s)
    def particular_solution_sleeper(self):
        # Timoshenko Particular solution when w_s = w_p 1 :
        #   w'''' + 0* w''' - k_pd/kAG*w'' + k_pd/EI*(w-w_s) = 0 
        # Assumption on particular: w(x)= ......+1+x
        (alpha,beta) = self.alpha,self.beta
        L = self.L        
        EIkAG =self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)
        kdEIkAg2  = self.padproperties.k_pd*self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)**2
        g1_0 = 1 #= np.np.cosh(beta*x)*np.cos(alpha*x)
        g2_0 = 0 #= np.cosh(beta*x)*np.sin(alpha*x)
        g3_0 = 0 #= np.sinh(beta*x)*np.cos(alpha*x)
        g4_0 = 0 #= np.sinh(beta*x)*np.sin(alpha*x)
        g1_L = np.cosh(beta*L)*np.cos(alpha*L) #= np.cosh(beta*x)*np.cos(alpha*x)
        g2_L = np.cosh(beta*L)*np.sin(alpha*L) #= np.cosh(beta*x)*np.sin(alpha*x)
        g3_L = np.sinh(beta*L)*np.cos(alpha*L) #= np.sinh(beta*x)*np.cos(alpha*x)
        g4_L = np.sinh(beta*L)*np.sin(alpha*L) #= np.sinh(beta*x)*np.sin(alpha*x)
        # Substituting in theta:
        #   theta = w' * (1- k_d*EI/(kAG)^2) +EI/(kAG)*w'''
        #   theta(x) = C_1 h_1 (x) + C_2 h_2 (x) + C_3 h_3 (x) + C_4 h_4 (x)
        
        h1_0 = 0 
        h2_0 = EIkAG*(3*alpha*beta**2 - alpha**3) - alpha*(kdEIkAg2 - 1) 
        h3_0 = - beta*(kdEIkAg2 - 1) - EIkAG*(3*alpha**2*beta - beta**3)
        h4_0 = 0
        h1_L = EIkAG*(alpha**3*np.cosh(L*beta)*np.sin(L*alpha) + beta**3*np.cos(L*alpha)*np.sinh(L*beta) - 3*alpha*beta**2*np.cosh(L*beta)*np.sin(L*alpha) - 3*alpha**2*beta*np.cos(L*alpha)*np.sinh(L*beta)) + (alpha*np.cosh(L*beta)*np.sin(L*alpha) - beta*np.cos(L*alpha)*np.sinh(L*beta))*(kdEIkAg2 - 1)
        h2_L = - EIkAG*(alpha**3*np.cos(L*alpha)*np.cosh(L*beta) - beta**3*np.sin(L*alpha)*np.sinh(L*beta) - 3*alpha*beta**2*np.cos(L*alpha)*np.cosh(L*beta) + 3*alpha**2*beta*np.sin(L*alpha)*np.sinh(L*beta)) - (alpha*np.cos(L*alpha)*np.cosh(L*beta) + beta*np.sin(L*alpha)*np.sinh(L*beta))*(kdEIkAg2 - 1)
        h3_L = EIkAG*(beta**3*np.cos(L*alpha)*np.cosh(L*beta) + alpha**3*np.sin(L*alpha)*np.sinh(L*beta) - 3*alpha**2*beta*np.cos(L*alpha)*np.cosh(L*beta) - 3*alpha*beta**2*np.sin(L*alpha)*np.sinh(L*beta)) - (beta*np.cos(L*alpha)*np.cosh(L*beta) - alpha*np.sin(L*alpha)*np.sinh(L*beta))*(kdEIkAg2 - 1)
        h4_L = - EIkAG*(alpha**3*np.cos(L*alpha)*np.sinh(L*beta) - beta**3*np.cosh(L*beta)*np.sin(L*alpha) - 3*alpha*beta**2*np.cos(L*alpha)*np.sinh(L*beta) + 3*alpha**2*beta*np.cosh(L*beta)*np.sin(L*alpha)) - (alpha*np.cos(L*alpha)*np.sinh(L*beta) + beta*np.cosh(L*beta)*np.sin(L*alpha))*(kdEIkAg2 - 1)
        
        GH = np.array([[g1_0, g2_0, g3_0, g4_0],
                      [h1_0, h2_0, h3_0, h4_0],
                      [g1_L, g2_L, g3_L, g4_L],
                      [h1_L, h2_L, h3_L, h4_L]])   
        N_ws_EF = np.linalg.solve(GH,[-1,0,-1,0])
        return N_ws_EF
    def w(self,xi,idx):
        (C1,C2,C3,C4) = self.coeffs[idx]
        return (np.cosh(self.beta*xi*self.L)*(C1*np.cos(self.alpha*xi*self.L)+C2*np.sin(self.alpha*xi*self.L))+
                np.sinh(self.beta*xi*self.L)*(C3*np.cos(self.alpha*xi*self.L)+C4*np.sin(self.alpha*xi*self.L)))
    def theta(self,xi,idx):
        (C1,C2,C3,C4) = self.coeffs[idx]
        EIkAG =self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)
        kdEIkAg2  = self.padproperties.k_pd*self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)**2
        
        theta1 = (EIkAG*(np.cosh(self.beta*xi*self.L)*(C1*self.alpha**3*np.sin(self.alpha*xi*self.L) - C2*self.alpha**3*np.cos(self.alpha*xi*self.L)) + 
                        np.sinh(self.beta*xi*self.L)*(C3*self.alpha**3*np.sin(self.alpha*xi*self.L) - C4*self.alpha**3*np.cos(self.alpha*xi*self.L)) + 
                        3*self.beta**2*np.cosh(self.beta*xi*self.L)*(C2*self.alpha*np.cos(self.alpha*xi*self.L) - C1*self.alpha*np.sin(self.alpha*xi*self.L)) + 
                        3*self.beta**2*np.sinh(self.beta*xi*self.L)*(C4*self.alpha*np.cos(self.alpha*xi*self.L) - C3*self.alpha*np.sin(self.alpha*xi*self.L)) +
                        self.beta**3*np.cosh(self.beta*xi*self.L)*(C3*np.cos(self.alpha*xi*self.L) + C4*np.sin(self.alpha*xi*self.L)) + 
                        self.beta**3*np.sinh(self.beta*xi*self.L)*(C1*np.cos(self.alpha*xi*self.L) + C2*np.sin(self.alpha*xi*self.L)) - 
                        3*self.beta*np.cosh(self.beta*xi*self.L)*(C4*self.alpha**2*np.sin(self.alpha*xi*self.L) + C3*self.alpha**2*np.cos(self.alpha*xi*self.L)) - 
                        3*self.beta*np.sinh(self.beta*xi*self.L)*(C2*self.alpha**2*np.sin(self.alpha*xi*self.L) + C1*self.alpha**2*np.cos(self.alpha*xi*self.L))) -
                    (kdEIkAg2 - 1)*(np.cosh(self.beta*xi*self.L)*(C2*self.alpha*np.cos(self.alpha*xi*self.L) - C1*self.alpha*np.sin(self.alpha*xi*self.L)) + 
                                    np.sinh(self.beta*xi*self.L)*(C4*self.alpha*np.cos(self.alpha*xi*self.L) - C3*self.alpha*np.sin(self.alpha*xi*self.L)) + 
                                    self.beta*np.cosh(self.beta*xi*self.L)*(C3*np.cos(self.alpha*xi*self.L) + C4*np.sin(self.alpha*xi*self.L)) +
                                    self.beta*np.sinh(self.beta*xi*self.L)*(C1*np.cos(self.alpha*xi*self.L) + C2*np.sin(self.alpha*xi*self.L)))) 
        return theta1
        
    def N_w1(self,xi):
        return self.w(xi,0)
    def N_t1(self,xi):
        return self.w(xi,1)
    def N_w2(self,xi):
        return self.w(xi,2)
    def N_t2(self,xi):
        return self.w(xi,3)
    def N_s(self,xi):
        return self.w(xi,4)+1
    
    # def N_e(self,xi):
    #     N_e_tim4 = np.vstack([self.N_w1(xi), 
    #                 self.N_t1(xi), 
    #                 self.N_w2(xi), 
    #                 self.N_t2(xi)])
    #     return N_e_tim4  
    def kc_contribution_bedding(self):
        # S_k = k_d*integral(N_e_EF.T*N_e_EF,0,L)
        N_w1w1_i = self.L*integrate.quad(lambda x: self.N_w1(x)*self.N_w1(x),0,1)[1]
        N_t1t1_i = self.L*integrate.quad(lambda x: self.N_t1(x)*self.N_t1(x),0,1)[1] 
        N_w2w2_i = self.L*integrate.quad(lambda x: self.N_w2(x)*self.N_w2(x),0,1)[1]
        N_t2t2_i = self.L*integrate.quad(lambda x: self.N_t2(x)*self.N_t2(x),0,1)[1]
        
        N_w1w2_i = self.L*integrate.quad(lambda x: self.N_w1(x)*self.N_w2(x),0,1)[1]
        N_w1t1_i = self.L*integrate.quad(lambda x: self.N_w1(x)*self.N_t1(x),0,1)[1]
        N_w1t2_i = self.L*integrate.quad(lambda x: self.N_w1(x)*self.N_t2(x),0,1)[1]
        
        N_t1w2_i = self.L*integrate.quad(lambda x: self.N_t1(x)*self.N_w2(x),0,1)[1]
        N_t1t2_i = self.L*integrate.quad(lambda x: self.N_t1(x)*self.N_t2(x),0,1)[1]
        
        N_w2t2_i = self.L*integrate.quad(lambda x: self.N_w2(x)*self.N_t2(x),0,1)[1]
        
        S_k_e = self.padproperties.k_pd*np.array([[N_w1w1_i,N_w1t1_i,N_w1w2_i,N_w1t2_i],
                                                  [N_w1t1_i,N_t1t1_i,N_t1w2_i,N_t1t2_i],
                                                  [N_w1w2_i,N_t1w2_i,N_w2w2_i,N_w2t2_i],
                                                  [N_w1t2_i,N_t1t2_i,N_w2t2_i,N_t2t2_i]])
        S_c_e = self.padproperties.c_pd*np.array([[N_w1w1_i,N_w1t1_i,N_w1w2_i,N_w1t2_i],
                                                  [N_w1t1_i,N_t1t1_i,N_t1w2_i,N_t1t2_i],
                                                  [N_w1w2_i,N_t1w2_i,N_w2w2_i,N_w2t2_i],
                                                  [N_w1t2_i,N_t1t2_i,N_w2t2_i,N_t2t2_i]])
        return S_k_e, S_c_e
    
    def B_EFb_t(self,x_i,idx):
        (C1,C2,C3,C4) = self.coeffs[idx]
        EIkAG = self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)
        kdEIkAg2  = self.padproperties.k_pd*self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)**2
        x = x_i*self.L
        alpha,beta = self.alpha,self.beta
    
        B_EFb = ((kdEIkAg2 - 1)*(np.cosh(beta*x)*(C1*alpha**3*np.sin(alpha*x) - C2*alpha**3*np.cos(alpha*x)) + np.sinh(beta*x)*(C3*alpha**3*np.sin(alpha*x) - C4*alpha**3*np.cos(alpha*x)) + 3*beta**2*np.cosh(beta*x)*(C2*alpha*np.cos(alpha*x) - C1*alpha*np.sin(alpha*x)) + 3*beta**2*np.sinh(beta*x)*(C4*alpha*np.cos(alpha*x) - C3*alpha*np.sin(alpha*x)) + beta**3*np.cosh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + beta**3*np.sinh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) - 3*beta*np.cosh(beta*x)*(C4*alpha**2*np.sin(alpha*x) + C3*alpha**2*np.cos(alpha*x)) - 3*beta*np.sinh(beta*x)*(C2*alpha**2*np.sin(alpha*x) + C1*alpha**2*np.cos(alpha*x))) - EIkAG*(10*beta**2*np.cosh(beta*x)*(C1*alpha**3*np.sin(alpha*x) - C2*alpha**3*np.cos(alpha*x)) - np.sinh(beta*x)*(C3*alpha**5*np.sin(alpha*x) - C4*alpha**5*np.cos(alpha*x)) - np.cosh(beta*x)*(C1*alpha**5*np.sin(alpha*x) - C2*alpha**5*np.cos(alpha*x)) - 10*beta**3*np.cosh(beta*x)*(C4*alpha**2*np.sin(alpha*x) + C3*alpha**2*np.cos(alpha*x)) - 10*beta**3*np.sinh(beta*x)*(C2*alpha**2*np.sin(alpha*x) + C1*alpha**2*np.cos(alpha*x)) + 10*beta**2*np.sinh(beta*x)*(C3*alpha**3*np.sin(alpha*x) - C4*alpha**3*np.cos(alpha*x)) + 5*beta**4*np.cosh(beta*x)*(C2*alpha*np.cos(alpha*x) - C1*alpha*np.sin(alpha*x)) + 5*beta**4*np.sinh(beta*x)*(C4*alpha*np.cos(alpha*x) - C3*alpha*np.sin(alpha*x)) + beta**5*np.cosh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + beta**5*np.sinh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) + 5*beta*np.cosh(beta*x)*(C4*alpha**4*np.sin(alpha*x) + C3*alpha**4*np.cos(alpha*x)) + 5*beta*np.sinh(beta*x)*(C2*alpha**4*np.sin(alpha*x) + C1*alpha**4*np.cos(alpha*x))))*(kdEIkAg2 - 1) - EIkAG*((kdEIkAg2 - 1)*(10*beta**2*np.cosh(beta*x)*(C1*alpha**3*np.sin(alpha*x) - C2*alpha**3*np.cos(alpha*x)) - np.sinh(beta*x)*(C3*alpha**5*np.sin(alpha*x) - C4*alpha**5*np.cos(alpha*x)) - np.cosh(beta*x)*(C1*alpha**5*np.sin(alpha*x) - C2*alpha**5*np.cos(alpha*x)) - 10*beta**3*np.cosh(beta*x)*(C4*alpha**2*np.sin(alpha*x) + C3*alpha**2*np.cos(alpha*x)) - 10*beta**3*np.sinh(beta*x)*(C2*alpha**2*np.sin(alpha*x) + C1*alpha**2*np.cos(alpha*x)) + 10*beta**2*np.sinh(beta*x)*(C3*alpha**3*np.sin(alpha*x) - C4*alpha**3*np.cos(alpha*x)) + 5*beta**4*np.cosh(beta*x)*(C2*alpha*np.cos(alpha*x) - C1*alpha*np.sin(alpha*x)) + 5*beta**4*np.sinh(beta*x)*(C4*alpha*np.cos(alpha*x) - C3*alpha*np.sin(alpha*x)) + beta**5*np.cosh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + beta**5*np.sinh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) + 5*beta*np.cosh(beta*x)*(C4*alpha**4*np.sin(alpha*x) + C3*alpha**4*np.cos(alpha*x)) + 5*beta*np.sinh(beta*x)*(C2*alpha**4*np.sin(alpha*x) + C1*alpha**4*np.cos(alpha*x))) - EIkAG*(np.cosh(beta*x)*(C1*alpha**7*np.sin(alpha*x) - C2*alpha**7*np.cos(alpha*x)) + np.sinh(beta*x)*(C3*alpha**7*np.sin(alpha*x) - C4*alpha**7*np.cos(alpha*x)) + 35*beta**4*np.cosh(beta*x)*(C1*alpha**3*np.sin(alpha*x) - C2*alpha**3*np.cos(alpha*x)) - 21*beta**2*np.cosh(beta*x)*(C1*alpha**5*np.sin(alpha*x) - C2*alpha**5*np.cos(alpha*x)) - 21*beta**5*np.cosh(beta*x)*(C4*alpha**2*np.sin(alpha*x) + C3*alpha**2*np.cos(alpha*x)) + 35*beta**3*np.cosh(beta*x)*(C4*alpha**4*np.sin(alpha*x) + C3*alpha**4*np.cos(alpha*x)) - 21*beta**5*np.sinh(beta*x)*(C2*alpha**2*np.sin(alpha*x) + C1*alpha**2*np.cos(alpha*x)) + 35*beta**3*np.sinh(beta*x)*(C2*alpha**4*np.sin(alpha*x) + C1*alpha**4*np.cos(alpha*x)) + 35*beta**4*np.sinh(beta*x)*(C3*alpha**3*np.sin(alpha*x) - C4*alpha**3*np.cos(alpha*x)) - 21*beta**2*np.sinh(beta*x)*(C3*alpha**5*np.sin(alpha*x) - C4*alpha**5*np.cos(alpha*x)) + 7*beta**6*np.cosh(beta*x)*(C2*alpha*np.cos(alpha*x) - C1*alpha*np.sin(alpha*x)) + 7*beta**6*np.sinh(beta*x)*(C4*alpha*np.cos(alpha*x) - C3*alpha*np.sin(alpha*x)) + beta**7*np.cosh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + beta**7*np.sinh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) - 7*beta*np.cosh(beta*x)*(C4*alpha**6*np.sin(alpha*x) + C3*alpha**6*np.cos(alpha*x)) - 7*beta*np.sinh(beta*x)*(C2*alpha**6*np.sin(alpha*x) + C1*alpha**6*np.cos(alpha*x))))
 
        return B_EFb
    def B_EFb_w(self,x_i,idx):
        (C1,C2,C3,C4) = self.coeffs[idx]
        EIkAG = self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)
        kdEIkAg2  = self.padproperties.k_pd*self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)**2
        x = x_i*self.L
        alpha,beta = self.alpha,self.beta

        B_EFb = EIkAG*(np.cosh(beta*x)*(C2*alpha**4*np.sin(alpha*x) + C1*alpha**4*np.cos(alpha*x)) + np.sinh(beta*x)*(C4*alpha**4*np.sin(alpha*x) + C3*alpha**4*np.cos(alpha*x)) - 6*beta**2*np.cosh(beta*x)*(C2*alpha**2*np.sin(alpha*x) + C1*alpha**2*np.cos(alpha*x)) - 6*beta**2*np.sinh(beta*x)*(C4*alpha**2*np.sin(alpha*x) + C3*alpha**2*np.cos(alpha*x)) + 4*beta**3*np.cosh(beta*x)*(C4*alpha*np.cos(alpha*x) - C3*alpha*np.sin(alpha*x)) + 4*beta**3*np.sinh(beta*x)*(C2*alpha*np.cos(alpha*x) - C1*alpha*np.sin(alpha*x)) + beta**4*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) + beta**4*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + 4*beta*np.cosh(beta*x)*(C3*alpha**3*np.sin(alpha*x) - C4*alpha**3*np.cos(alpha*x)) + 4*beta*np.sinh(beta*x)*(C1*alpha**3*np.sin(alpha*x) - C2*alpha**3*np.cos(alpha*x))) - (kdEIkAg2 - 1)*(2*beta*np.cosh(beta*x)*(C4*alpha*np.cos(alpha*x) - C3*alpha*np.sin(alpha*x)) - np.sinh(beta*x)*(C4*alpha**2*np.sin(alpha*x) + C3*alpha**2*np.cos(alpha*x)) - np.cosh(beta*x)*(C2*alpha**2*np.sin(alpha*x) + C1*alpha**2*np.cos(alpha*x)) + 2*beta*np.sinh(beta*x)*(C2*alpha*np.cos(alpha*x) - C1*alpha*np.sin(alpha*x)) + beta**2*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) + beta**2*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)))
        return B_EFb
    
    def B_EFsh_t(self,x_i,idx):
        (C1,C2,C3,C4) = self.coeffs[idx]
        x = x_i*self.L
        alpha,beta = self.alpha,self.beta
        kdkAg = self.padproperties.k_pd/(self.railproperties.k*self.railproperties.A*self.railproperties.G)
        EIkAG = self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)
        kdEIkAg2  = self.padproperties.k_pd*self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)**2
        
        B_EFsh = EIkAG*((kdEIkAg2 - 1)*(alpha**4*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) + beta**4*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) + alpha**4*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + beta**4*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + 4*alpha*beta**3*np.cosh(beta*x)*(C4*np.cos(alpha*x) - C3*np.sin(alpha*x)) - 4*alpha**3*beta*np.cosh(beta*x)*(C4*np.cos(alpha*x) - C3*np.sin(alpha*x)) + 4*alpha*beta**3*np.sinh(beta*x)*(C2*np.cos(alpha*x) - C1*np.sin(alpha*x)) - 4*alpha**3*beta*np.sinh(beta*x)*(C2*np.cos(alpha*x) - C1*np.sin(alpha*x)) - 6*alpha**2*beta**2*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) - 6*alpha**2*beta**2*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x))) - EIkAG*(beta**6*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) - alpha**6*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) - alpha**6*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + beta**6*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + 6*alpha*beta**5*np.cosh(beta*x)*(C4*np.cos(alpha*x) - C3*np.sin(alpha*x)) + 6*alpha**5*beta*np.cosh(beta*x)*(C4*np.cos(alpha*x) - C3*np.sin(alpha*x)) + 6*alpha*beta**5*np.sinh(beta*x)*(C2*np.cos(alpha*x) - C1*np.sin(alpha*x)) + 6*alpha**5*beta*np.sinh(beta*x)*(C2*np.cos(alpha*x) - C1*np.sin(alpha*x)) - 15*alpha**2*beta**4*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) + 15*alpha**4*beta**2*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) - 20*alpha**3*beta**3*np.cosh(beta*x)*(C4*np.cos(alpha*x) - C3*np.sin(alpha*x)) - 20*alpha**3*beta**3*np.sinh(beta*x)*(C2*np.cos(alpha*x) - C1*np.sin(alpha*x)) - 15*alpha**2*beta**4*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + 15*alpha**4*beta**2*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x))) + kdkAg*(EIkAG*(alpha**4*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) + beta**4*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) + alpha**4*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + beta**4*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + 4*alpha*beta**3*np.cosh(beta*x)*(C4*np.cos(alpha*x) - C3*np.sin(alpha*x)) - 4*alpha**3*beta*np.cosh(beta*x)*(C4*np.cos(alpha*x) - C3*np.sin(alpha*x)) + 4*alpha*beta**3*np.sinh(beta*x)*(C2*np.cos(alpha*x) - C1*np.sin(alpha*x)) - 4*alpha**3*beta*np.sinh(beta*x)*(C2*np.cos(alpha*x) - C1*np.sin(alpha*x)) - 6*alpha**2*beta**2*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) - 6*alpha**2*beta**2*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x))) - (kdEIkAg2 - 1)*(beta**2*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) - alpha**2*np.cosh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) - alpha**2*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + beta**2*np.sinh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + 2*alpha*beta*np.cosh(beta*x)*(C4*np.cos(alpha*x) - C3*np.sin(alpha*x)) + 2*alpha*beta*np.sinh(beta*x)*(C2*np.cos(alpha*x) - C1*np.sin(alpha*x)))))
        return B_EFsh
    def B_EFsh_w(self,x_i,idx):
        (C1,C2,C3,C4) = self.coeffs[idx]
        x = x_i*self.L
        alpha,beta = self.alpha,self.beta
        kdkAg = self.padproperties.k_pd/(self.railproperties.k*self.railproperties.A*self.railproperties.G)
        EIkAG = self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)
    
        B_EFsh = EIkAG*(kdkAg*(alpha*np.cosh(beta*x)*(C2*np.cos(alpha*x) - C1*np.sin(alpha*x)) + beta*np.cosh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + alpha*np.sinh(beta*x)*(C4*np.cos(alpha*x) - C3*np.sin(alpha*x)) + beta*np.sinh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x))) + alpha**3*np.cosh(beta*x)*(C2*np.cos(alpha*x) - C1*np.sin(alpha*x)) - beta**3*np.cosh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + alpha**3*np.sinh(beta*x)*(C4*np.cos(alpha*x) - C3*np.sin(alpha*x)) - beta**3*np.sinh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) - 3*alpha*beta**2*np.cosh(beta*x)*(C2*np.cos(alpha*x) - C1*np.sin(alpha*x)) + 3*alpha**2*beta*np.cosh(beta*x)*(C3*np.cos(alpha*x) + C4*np.sin(alpha*x)) + 3*alpha**2*beta*np.sinh(beta*x)*(C1*np.cos(alpha*x) + C2*np.sin(alpha*x)) - 3*alpha*beta**2*np.sinh(beta*x)*(C4*np.cos(alpha*x) - C3*np.sin(alpha*x)))
        return B_EFsh

    def N_EF_w(self,x_i,idx):
        return self.w(x_i,idx)
    def N_EFb_w(self,x_i,idx):
        return self.theta(x_i,idx)
    
    def B_EFsh2(self,x_i,idx1,idx2):
        # Nodal Displacement to Shear Strain 
        B_EFsh = self.B_EFsh_w(x_i,idx1)*self.B_EFsh_w(x_i,idx2)
        return B_EFsh
    def B_EFb2(self,x_i,idx1,idx2):
        # Nodal Displacement to Bending Strain 
        B_EFb = self.B_EFb_w(x_i,idx1)*self.B_EFb_w(x_i,idx2)
        return B_EFb
    
    def N_EF2(self,x_i,idx1,idx2):
        # Nodal Displacement mass contrib 
        N_EF = self.N_EF_w(x_i,idx1)*self.N_EF_w(x_i,idx2)
        return N_EF
    def N_EFb2(self,x_i,idx1,idx2):
        # Nodal Displacement mass contrib 
        N_EFb = self.N_EFb_w(x_i,idx1)*self.N_EFb_w(x_i,idx2)
        return N_EFb

    def stiffnessMatrixTEEF(self):
        i1,i2 = np.triu_indices(5)
        K_ef = np.zeros((5,5))
        S_ef = np.zeros((5,5))
        for i in range(len(i1)):
            idx1 = i1[i]
            idx2 = i2[i]
            K_ef12 = (self.railproperties.E*self.railproperties.I*self.L*integrate.quad(self.B_EFb2,0,1, args=(idx1,idx2))[0] + 
                        self.railproperties.G*self.railproperties.A*self.railproperties.k*self.L*integrate.quad(self.B_EFsh2,0,1, args=(idx1,idx2))[0])                       
            S_ef12 = (self.padproperties.k_pd*self.L*integrate.quad(self.N_EF2,0,1, args=(idx1,idx2))[0])
            
            K_ef[idx1,idx2] = K_ef12 
            K_ef[idx2,idx1] = K_ef12 
            S_ef[idx1,idx2] = S_ef12 
            S_ef[idx2,idx1] = S_ef12 
        self.K_EF = S_ef +K_ef
        return K_ef+S_ef

    def massMatrixTEEF(self):
        i1,i2 = np.triu_indices(5)
        M_ef = np.zeros((5,5))
        for i in range(len(i1)):
            idx1 = i1[i]
            idx2 = i2[i]
            M_ef12 = (self.railproperties.rho*self.railproperties.A*self.L*integrate.quad(self.N_EF2,0,1, args=(idx1,idx2))[0] +
                self.railproperties.rho*self.railproperties.I*self.L*integrate.quad(self.N_EFb2,0,1, args=(idx1,idx2))[0])
            # print((self.railproperties.rho*self.railproperties.A*self.L*integrate.quad(self.N_EF2,0,1, args=(idx1,idx2))[0],
            #        self.railproperties.rho*self.railproperties.I*self.L*integrate.quad(self.N_EFb2,0,1, args=(idx1,idx2))[0]))
            M_ef[idx1,idx2] = M_ef12
            M_ef[idx2,idx1] = M_ef12
        self.M_EF = M_ef
        return M_ef
        
    def dampingMatrixTEEF(self):
        i1,i2 = np.triu_indices(5)
        D_ef = np.zeros((5,5))
        for i in range(len(i1)):
            idx1 = i1[i]
            idx2 = i2[i]
            D_ef12 = (self.padproperties.c_pd*self.L*integrate.quad(self.N_EF2,0,1, args=(idx1,idx2))[0])
            D_ef[idx1,idx2] = D_ef12 
            D_ef[idx2,idx1] = D_ef12 
        self.D_EF = D_ef
        return D_ef 
    def displacement_field_TIM4eb(self,xi,coeffs):
        (C1,C2,C3,C4) = coeffs
        # Displacement field on left side:
        w = (np.cosh(self.beta*xi*self.L)*(C1*np.cos(self.alpha*xi*self.L)+C2*np.sin(self.alpha*xi*self.L))+
                np.sinh(self.beta*xi*self.L)*(C3*np.cos(self.alpha*xi*self.L)+C4*np.sin(self.alpha*xi*self.L)))
        return w
    def rotation_field_TIM4eb(self,xi,coeffs):
        (C1,C2,C3,C4) = coeffs
        EIkAG =self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)
        kdEIkAg2  = self.padproperties.k_pd*self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)**2
        
        theta = (EIkAG*(np.cosh(self.beta*xi*self.L)*(C1*self.alpha**3*np.sin(self.alpha*xi*self.L) - C2*self.alpha**3*np.cos(self.alpha*xi*self.L)) + 
                        np.sinh(self.beta*xi*self.L)*(C3*self.alpha**3*np.sin(self.alpha*xi*self.L) - C4*self.alpha**3*np.cos(self.alpha*xi*self.L)) + 
                        3*self.beta**2*np.cosh(self.beta*xi*self.L)*(C2*self.alpha*np.cos(self.alpha*xi*self.L) - C1*self.alpha*np.sin(self.alpha*xi*self.L)) + 
                        3*self.beta**2*np.sinh(self.beta*xi*self.L)*(C4*self.alpha*np.cos(self.alpha*xi*self.L) - C3*self.alpha*np.sin(self.alpha*xi*self.L)) +
                        self.beta**3*np.cosh(self.beta*xi*self.L)*(C3*np.cos(self.alpha*xi*self.L) + C4*np.sin(self.alpha*xi*self.L)) + 
                        self.beta**3*np.sinh(self.beta*xi*self.L)*(C1*np.cos(self.alpha*xi*self.L) + C2*np.sin(self.alpha*xi*self.L)) - 
                        3*self.beta*np.cosh(self.beta*xi*self.L)*(C4*self.alpha**2*np.sin(self.alpha*xi*self.L) + C3*self.alpha**2*np.cos(self.alpha*xi*self.L)) - 
                        3*self.beta*np.sinh(self.beta*xi*self.L)*(C2*self.alpha**2*np.sin(self.alpha*xi*self.L) + C1*self.alpha**2*np.cos(self.alpha*xi*self.L))) -
                    (kdEIkAg2 - 1)*(np.cosh(self.beta*xi*self.L)*(C2*self.alpha*np.cos(self.alpha*xi*self.L) - C1*self.alpha*np.sin(self.alpha*xi*self.L)) + 
                                    np.sinh(self.beta*xi*self.L)*(C4*self.alpha*np.cos(self.alpha*xi*self.L) - C3*self.alpha*np.sin(self.alpha*xi*self.L)) + 
                                    self.beta*np.cosh(self.beta*xi*self.L)*(C3*np.cos(self.alpha*xi*self.L) + C4*np.sin(self.alpha*xi*self.L)) +
                                    self.beta*np.sinh(self.beta*xi*self.L)*(C1*np.cos(self.alpha*xi*self.L) + C2*np.sin(self.alpha*xi*self.L)))) 
        return theta
    def Moment(self,xi,coeffs):
        # Moment from Timoshenko governing equations:
        # -EI*theta'
        (C1,C2,C3,C4) = coeffs
        alpha = self.alpha
        beta = self.beta
        EIkAG =self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)
        kdEIkAg2  = self.padproperties.k_pd*self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)**2
        
        M_xi = -self.railproperties.E*self.railproperties.I*(EIkAG*(alpha**4*np.cosh(beta*xi*self.L)*(C1*np.cos(alpha*xi*self.L) + C2*np.sin(alpha*xi*self.L)) + 
                           beta**4*np.cosh(beta*xi*self.L)*(C1*np.cos(alpha*xi*self.L) + C2*np.sin(alpha*xi*self.L)) + 
                           alpha**4*np.sinh(beta*xi*self.L)*(C3*np.cos(alpha*xi*self.L) + C4*np.sin(alpha*xi*self.L)) + 
                           beta**4*np.sinh(beta*xi*self.L)*(C3*np.cos(alpha*xi*self.L) + C4*np.sin(alpha*xi*self.L)) + 
                           4*alpha*beta**3*np.cosh(beta*xi*self.L)*(C4*np.cos(alpha*xi*self.L) - C3*np.sin(alpha*xi*self.L)) - 
                           4*alpha**3*beta*np.cosh(beta*xi*self.L)*(C4*np.cos(alpha*xi*self.L) - C3*np.sin(alpha*xi*self.L)) + 
                           4*alpha*beta**3*np.sinh(beta*xi*self.L)*(C2*np.cos(alpha*xi*self.L) - C1*np.sin(alpha*xi*self.L)) - 
                           4*alpha**3*beta*np.sinh(beta*xi*self.L)*(C2*np.cos(alpha*xi*self.L) - C1*np.sin(alpha*xi*self.L)) - 
                           6*alpha**2*beta**2*np.cosh(beta*xi*self.L)*(C1*np.cos(alpha*xi*self.L) + C2*np.sin(alpha*xi*self.L)) - 
                           6*alpha**2*beta**2*np.sinh(beta*xi*self.L)*(C3*np.cos(alpha*xi*self.L) + C4*np.sin(alpha*xi*self.L))) - 
                    (kdEIkAg2 - 1)*(beta**2*np.cosh(beta*xi*self.L)*(C1*np.cos(alpha*xi*self.L) + C2*np.sin(alpha*xi*self.L)) -
                                    alpha**2*np.cosh(beta*xi*self.L)*(C1*np.cos(alpha*xi*self.L) + C2*np.sin(alpha*xi*self.L)) - 
                                    alpha**2*np.sinh(beta*xi*self.L)*(C3*np.cos(alpha*xi*self.L) + C4*np.sin(alpha*xi*self.L)) + 
                                    beta**2*np.sinh(beta*xi*self.L)*(C3*np.cos(alpha*xi*self.L) + C4*np.sin(alpha*xi*self.L)) + 
                                    2*alpha*beta*np.cosh(beta*xi*self.L)*(C4*np.cos(alpha*xi*self.L) - C3*np.sin(alpha*xi*self.L)) + 
                                    2*alpha*beta*np.sinh(beta*xi*self.L)*(C2*np.cos(alpha*xi*self.L) - C1*np.sin(alpha*xi*self.L))))
        return M_xi
    def Shear(self,xi,coeffs):
        # Moment from Timoshenko governing equations:
        # -EI*theta'
        (C1,C2,C3,C4) = coeffs
        alpha = self.alpha
        beta = self.beta
        EIkAG =self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)
        kdEIkAg2  = self.padproperties.k_pd*self.railproperties.E*self.railproperties.I/(self.railproperties.k*self.railproperties.A*self.railproperties.G)**2
        
        V_xi = (self.railproperties.k*self.railproperties.A*self.railproperties.G)*((kdEIkAg2 - 1)*(np.cosh(beta*xi*self.L)*(C2*alpha*np.cos(alpha*xi*self.L) - 
                    C1*alpha*np.sin(alpha*xi*self.L)) + np.sinh(beta*xi*self.L)*(C4*alpha*np.cos(alpha*xi*self.L) - 
                    C3*alpha*np.sin(alpha*xi*self.L)) + beta*np.cosh(beta*xi*self.L)*(C3*np.cos(alpha*xi*self.L) + 
                    C4*np.sin(alpha*xi*self.L)) + beta*np.sinh(beta*xi*self.L)*(C1*np.cos(alpha*xi*self.L) + 
                    C2*np.sin(alpha*xi*self.L))) + np.cosh(beta*xi*self.L)*(C2*alpha*np.cos(alpha*xi*self.L) - 
                    C1*alpha*np.sin(alpha*xi*self.L)) + np.sinh(beta*xi*self.L)*(C4*alpha*np.cos(alpha*xi*self.L) - 
                    C3*alpha*np.sin(alpha*xi*self.L)) - EIkAG*(np.cosh(beta*xi*self.L)*(C1*alpha**3*np.sin(alpha*xi*self.L) - 
                    C2*alpha**3*np.cos(alpha*xi*self.L)) + np.sinh(beta*xi*self.L)*(C3*alpha**3*np.sin(alpha*xi*self.L) - 
                    C4*alpha**3*np.cos(alpha*xi*self.L)) + 3*beta**2*np.cosh(beta*xi*self.L)*(C2*alpha*np.cos(alpha*xi*self.L) - 
                    C1*alpha*np.sin(alpha*xi*self.L)) + 3*beta**2*np.sinh(beta*xi*self.L)*(C4*alpha*np.cos(alpha*xi*self.L) - 
                    C3*alpha*np.sin(alpha*xi*self.L)) + beta**3*np.cosh(beta*xi*self.L)*(C3*np.cos(alpha*xi*self.L) + 
                    C4*np.sin(alpha*xi*self.L)) + beta**3*np.sinh(beta*xi*self.L)*(C1*np.cos(alpha*xi*self.L) + 
                    C2*np.sin(alpha*xi*self.L)) - 3*beta*np.cosh(beta*xi*self.L)*(C4*alpha**2*np.sin(alpha*xi*self.L) + 
                    C3*alpha**2*np.cos(alpha*xi*self.L)) - 3*beta*np.sinh(beta*xi*self.L)*(C2*alpha**2*np.sin(alpha*xi*self.L) + 
                    C1*alpha**2*np.cos(alpha*xi*self.L))) + beta*np.cosh(beta*xi*self.L)*(C3*np.cos(alpha*xi*self.L) + 
                    C4*np.sin(alpha*xi*self.L)) + beta*np.sinh(beta*xi*self.L)*(C1*np.cos(alpha*xi*self.L) + C2*np.sin(alpha*xi*self.L)))
                     
        return V_xi
    def point_load_flexibility_TIM4(self,xi):   
        # Calculated using a beam split into two sections , L and R, of length xi_L, xi_R
        if (xi-0.5)<10**-6:
            # Cannot solve a perfectly perfectly symmetric system, matrix solve error
            # Here precision 1.2e-7 is float32.
            xi = xi -10**-6
        xi_L = xi
        xi_R = 1-xi_L
        # Displacement field on left side:
        # w_L = (np.cosh(self.beta*xi*self.L)*(C1*np.cos(self.alpha*xi*self.L)+C2*np.sin(self.alpha*xi*self.L))+
        #         np.sinh(self.beta*xi*self.L)*(C3*np.cos(self.alpha*xi*self.L)+C4*np.sin(self.alpha*xi*self.L)))
        # Displacement field on right side:
        # w_R = (np.cosh(self.beta*xi*self.L)*(C5*np.cos(self.alpha*xi*self.L)+C6*np.sin(self.alpha*xi*self.L))+
        #         np.sinh(self.beta*xi*self.L)*(C7*np.cos(self.alpha*xi*self.L)+C8*np.sin(self.alpha*xi*self.L)))
        # Boundaries at xi_L = 0 and xi_R = 1-xi_L
        # w_L(0) = 0 left displacement C1,C2,C3,C4
        w_L_0 = self.displacement_field_TIM4eb(0,np.eye(4)).reshape(1,-1)
        # t_L(0) = 0 left displacement C1,C2,C3,C4
        t_L_0 = self.rotation_field_TIM4eb(0,np.eye(4)).reshape(1,-1)
        # w_L(end) = 0 left displacement C1,C2,C3,C4
        w_R_end = self.displacement_field_TIM4eb(xi_R,np.eye(4)).reshape(1,-1)
        # t_L(end) = 0 left displacement C1,C2,C3,C4    
        t_R_end =  self.rotation_field_TIM4eb(xi_R,np.eye(4)).reshape(1,-1)
        # Compatibility of displacement at the loading point:
        w_LR_xi = np.hstack((self.displacement_field_TIM4eb(xi,np.eye(4)).reshape(1,-1), self.displacement_field_TIM4eb(0,np.eye(4)).reshape(1,-1)))
        t_LR_xi = np.hstack((self.rotation_field_TIM4eb(xi,np.eye(4)).reshape(1,-1), self.rotation_field_TIM4eb(0,np.eye(4)).reshape(1,-1)))
        # Equilibrium of moment and shear at the loading point:
        m_LR_xi = np.hstack((-self.Moment(xi,np.eye(4)) , self.Moment(0,np.eye(4))))
        s_LR_xi = np.hstack((-self.Shear(xi,np.eye(4)) , -self.Shear(0,np.eye(4))))
        # Get the coefficients C1-C8 that fullfill the relations:
        KG = np.vstack((np.hstack((w_L_0,np.zeros((1,4)))),             # Vertical left node
             np.hstack((t_L_0,np.zeros((1,4)))),                        # Rotation left node
             np.hstack((w_LR_xi )),  # Vertical mid node
             np.hstack((t_LR_xi )),  # Rotation mid node
             np.hstack((np.zeros((1,4)),w_R_end)),                        # Vertical right node
             np.hstack((np.zeros((1,4)),t_R_end)),                        # Rotation right node
             m_LR_xi,                                                   # Moment Eq
             s_LR_xi))                                                  # Shear Eq  
        C18 = scipy.linalg.solve(KG,[0,0,0,0,0,0,0,-1])
        self.coeffs18 = C18
        # Calculate the displacement at the node:
        displ_mid = self.displacement_field_TIM4eb(xi,C18[:4])
        rot_mid =   self.rotation_field_TIM4eb(xi,C18[:4])
        return displ_mid#, rot_mid
    
    
    # def Delta(self,omega):
    #     rho = self.railproperties.rho
    #     k   = self.railproperties.k
    #     A   = self.railproperties.A
    #     G   = self.railproperties.G
    #     I   = self.railproperties.I
    #     E   = self.railproperties.E
    #     kd  = self.padproperties.k_pd
    #     delta = (- I*(k*A*G)*rho*omega**2 + (E*I)*kd)**2/((E*I)**2*(k*A*G)**2) + (4*(A*rho*omega**2 + kd)*(- I*rho*omega**2 + (k*A*G)))/((E*I)*(k*A*G))
    #     return delta
    # def DeltaZ(self,omega):
    #     rho = self.railproperties.rho
    #     k   = self.railproperties.k
    #     A   = self.railproperties.A
    #     G   = self.railproperties.G
    #     I   = self.railproperties.I
    #     E   = self.railproperties.E
    #     kd  = self.padproperties.k_pd
    #     delta = (- I*(k*A*G)*rho*omega**2 + (E*I)*kd)**2/((E*I)**2*(k*A*G)**2) + (4*(A*rho*omega**2 + kd)*(- I*rho*omega**2 + (k*A*G)))/((E*I)*(k*A*G))
    #     d = (I*k*A*G*rho*omega**2 - E*I*kd)/(E*I*k*A*G) 
    #     return np.sqrt(delta)-d
    # def Lambda(self,omega):
    #     rho = self.railproperties.rho
    #     k   = self.railproperties.k
    #     A   = self.railproperties.A
    #     G   = self.railproperties.G
    #     I   = self.railproperties.I
    #     E   = self.railproperties.E
    #     L   = self.L
    #     kd  = self.padproperties.k_pd
    #     omega_0 = np.sqrt(k*A*G/(rho*I)) # cut off frequency
    #     #Wavenumbers http://yadda.icm.edu.pl/yadda/element/bwmeta1.element.baztech-article-BWM4-0019-0012/c/httpwww_ptmts_org_plmajkut-1-09.pdf Majkut L. Free and forced vibrations of Timoshenko beams described by single difference equation. J Theor Appl Mech. 2009;47(1):193–210.
    #     # Solution of free vibration equations of beam on elastic soil by using differential transform method https://www.sciencedirect.com/science/article/pii/S0307904X07001357
     
    #     d = (I*k*A*G*rho*omega**2 - E*I*kd)/(E*I*k*A*G) 
    #     delta = (- I*(k*A*G)*rho*omega**2 + (E*I)*kd)**2/((E*I)**2*(k*A*G)**2) + (4*(A*rho*omega**2 + kd)*(- I*rho*omega**2 + (k*A*G)))/((E*I)*(k*A*G))
    #     if delta<0:
    #         raise Warning('not implemented')
    #         # Solution of free vibration equations of beam on elastic soil by using differential transform method https://www.sciencedirect.com/science/article/pii/S0307904X07001357
    #     if np.sqrt(delta)>d:
    #         print('a',omega,delta, -d+np.sqrt(delta) )             
    #         lambda_1 = np.sqrt((d + np.sqrt(delta))/2)
    #         lambda_2 = np.sqrt((-d + np.sqrt(delta))/2)
    #         Lambda = np.array([[1,0,1,0],
    #                      [ 0, -(- (k*A*G)**2*lambda_1 + (E*I)*(k*A*G)*lambda_1**3 + (E*I)*kd*lambda_1)/((k*A*G)**2 - I*rho*(k*A*G)*omega**2), 
    #                       0, ((k*A*G)**2*lambda_2 + (E*I)*(k*A*G)*lambda_2**3 - (E*I)*kd*lambda_2)/((k*A*G)**2 - I*rho*(k*A*G)*omega**2)]  ,
    #                      [np.cos(lambda_1*L),np.sin(lambda_1*L),np.cosh(lambda_2*L),np.sinh(lambda_2*L)],
    #                      [ ((E*I)*kd*lambda_1*np.sin(L*lambda_1) - (k*A*G)**2*lambda_1*np.sin(L*lambda_1) + (E*I)*(k*A*G)*lambda_1**3*np.sin(L*lambda_1))/((k*A*G)**2 - I*rho*(k*A*G)*omega**2),
    #                       -((E*I)*(k*A*G)*lambda_1**3*np.cos(L*lambda_1) - (k*A*G)**2*lambda_1*np.cos(L*lambda_1) + (E*I)*kd*lambda_1*np.cos(L*lambda_1))/((k*A*G)**2 - I*rho*(k*A*G)*omega**2), 
    #                       ((k*A*G)**2*lambda_2*np.sinh(L*lambda_2) - (E*I)*kd*lambda_2*np.sinh(L*lambda_2) + (E*I)*(k*A*G)*lambda_2**3*np.sinh(L*lambda_2))/((k*A*G)**2 - I*rho*(k*A*G)*omega**2), 
    #                       ((k*A*G)**2*lambda_2*np.cosh(L*lambda_2) + (E*I)*(k*A*G)*lambda_2**3*np.cosh(L*lambda_2) - (E*I)*kd*lambda_2*np.cosh(L*lambda_2))/((k*A*G)**2 - I*rho*(k*A*G)*omega**2)]])
            
    #     elif np.sqrt(delta)<d:
    #         print('b',omega,delta, d-np.sqrt(delta) )  
    #         lambda_1 = np.sqrt((d + np.sqrt(delta))/2)
    #         lambda_3 = np.sqrt((d - np.sqrt(delta))/2)
    #         Lambda = np.array([[1,0,1,0],
    #                            [ 0, -(- (k*A*G)**2*lambda_1 + (E*I)*(k*A*G)*lambda_1**3 + (E*I)*kd*lambda_1)/((k*A*G)**2 - I*rho*(k*A*G)*omega**2), 
    #                             0, -(- (k*A*G)**2*lambda_3 + (E*I)*(k*A*G)*lambda_3**3 + (E*I)*kd*lambda_3)/((k*A*G)**2 - I*rho*(k*A*G)*omega**2)],
    #                           [np.cos(lambda_1*L),np.sin(lambda_1*L),np.cos(lambda_3*L),np.sin(lambda_3*L)],
    #                           [ ((E*I)*kd*lambda_1*np.sin(L*lambda_1) - (k*A*G)**2*lambda_1*np.sin(L*lambda_1) + (E*I)*(k*A*G)*lambda_1**3*np.sin(L*lambda_1))/((k*A*G)**2 - I*rho*(k*A*G)*omega**2),
    #                            -((E*I)*(k*A*G)*lambda_1**3*np.cos(L*lambda_1) - (k*A*G)**2*lambda_1*np.cos(L*lambda_1) + (E*I)*kd*lambda_1*np.cos(L*lambda_1))/((k*A*G)**2 - I*rho*(k*A*G)*omega**2), 
    #                            ((E*I)*kd*lambda_3*np.sin(L*lambda_3) - (k*A*G)**2*lambda_3*np.sin(L*lambda_3) + (E*I)*(k*A*G)*lambda_3**3*np.sin(L*lambda_3))/((k*A*G)**2 - I*rho*(k*A*G)*omega**2), 
    #                            -((E*I)*(k*A*G)*lambda_3**3*np.cos(L*lambda_3) - (k*A*G)**2*lambda_3*np.cos(L*lambda_3) + (E*I)*kd*lambda_3*np.cos(L*lambda_3))/((k*A*G)**2 - I*rho*(k*A*G)*omega**2)]])

    #     else:
    #         raise Warning('no compatible omega, omega = {}, omega_0 = {}'.format(omega,omega_0))
             
    #     return Lambda
    
    def Lambda(self,omega):
        rho = self.railproperties.rho
        k   = self.railproperties.k
        A   = self.railproperties.A
        G   = self.railproperties.G
        I   = self.railproperties.I
        E   = self.railproperties.E
        L   = self.L
        omega_0 = np.sqrt(k*A*G/(rho*I)) # cut off frequency
        #Wavenumbers http://yadda.icm.edu.pl/yadda/element/bwmeta1.element.baztech-article-BWM4-0019-0012/c/httpwww_ptmts_org_plmajkut-1-09.pdf Majkut L. Free and forced vibrations of Timoshenko beams described by single difference equation. J Theor Appl Mech. 2009;47(1):193–210.
     
        p = omega**2*rho/(k*G)+k*A*G/(E*I)
        d = omega**2*rho*I*(1+E/(G*k))/(E*I)
        e = omega**2*(omega**2*rho**2*I/(G*k)-rho*A)/(E*I)
        delta = d**2-4*e
      
        if omega<omega_0:               
            lambda_1 = np.sqrt((d + np.sqrt(delta))/2)
            lambda_2 = np.sqrt((-d + np.sqrt(delta))/2)
            Lambda = np.array([[1,0,1,0],
                         [0,lambda_1*(p-lambda_1**2),0,lambda_2*(p+lambda_2**2)],
                         [np.cos(lambda_1*L),np.sin(lambda_1*L),np.cosh(lambda_2*L),np.sinh(lambda_2*L)],
                         [np.sin(lambda_1*L)*(lambda_1**3-p*lambda_1),np.cos(lambda_1*L)*(-lambda_1**3+p*lambda_1),
                          np.sinh(lambda_2*L)*(lambda_2**3+p*lambda_2),np.cosh(lambda_2*L)*(lambda_2**3+p*lambda_2)]])
        elif omega > omega_0:
            lambda_1 = np.sqrt((d + np.sqrt(delta))/2)
            lambda_2 = np.sqrt((d - np.sqrt(delta))/2)
            Lambda = np.array([[1,0,1,0],
                              [0,lambda_1*(p-lambda_1**2),0,lambda_2*(p-lambda_2**2)],
                              [np.cos(lambda_1*L),np.sin(lambda_1*L),np.cos(lambda_2*L),np.sin(lambda_2*L)],
                              [np.sin(lambda_1*L)*(lambda_1**3-p*lambda_1),np.cos(lambda_1*L)*(-lambda_1**3+p*lambda_1),
                               np.sin(lambda_2*L)*(lambda_2**3-p*lambda_2),np.cos(lambda_2*L)*(-lambda_2**3+p*lambda_2)]])
        else:
            raise Warning('no compatible omega, omega = {}, omega_0 = {}'.format(omega,omega_0))
             
        return Lambda
    def DetLambda(self,omega):
        return np.linalg.det(self.Lambda(omega))
    def CalcP(self,omega):
        # Compute the coefficients of P
        Lambda_o = self.Lambda(omega)
        # np.det(Lamda)
        firstcol = Lambda_o[:,0].copy()
        # we impose a condition that first term be 1, 
        x = np.linalg.lstsq(Lambda_o[:, 1:], -firstcol,rcond = None)[0]
        x = np.r_[1, x]
        # Normalize them by integral(rho*A*W**2+rho*I*Theta**2,0,L)
        P = self.NormalizeP(x,omega)
        return P        
    def NormalizeP(self,P,omega):
        rho = self.railproperties.rho
        A   = self.railproperties.A
        I   = self.railproperties.I
        norm = self.L*integrate.quad(lambda x: rho*A*self.W(x,P,omega)**2 +rho*I*self.T(x,P,omega)**2,0,1)[0]
        
        # scipy.optimize.brentq(lambda x: integrate.quad(lambda x: rho*A*self.W(x,P,omega)**2 +rho*I*self.T(x,P,omega)**2,0,1)[0]-1, 
        
        P /= norm**0.5
        return P
    def preprocess(self,f,xmin,xmax,step,args=()):  
        if not isinstance(args, tuple):
                args = (args,)
        # Find when the function changes sign. Subdivide
        first_sign = f(xmin) > 0 # True if f(xmin) > 0, otherwise False
        x = xmin + step
        while x <= xmax: # This loop detects when the function changes its sign
            fstep = f(x,*args)
            if first_sign and fstep < 0:
                return x
            elif not(first_sign) and fstep > 0:
                return x
            x += step
        return x # If you ever reach here, that means that there isn't a zero in the function    
    def subdividespace(self,f,xmin,xmax,step,args=()):   
        if not isinstance(args, tuple):
                args = (args,)
        x_list = []
        x = xmin
        while x < xmax: # This discovers when f changes its sign
            x_list.append(x)
            x = self.preprocess(f,x,xmax,step,args)
        # x_list.append(xmax)
        return x_list
    def GetNaturalFrequencies(self,n=5):
        # Gets the first n natural vibration frequencies of timoshenko beam.
        rho = self.railproperties.rho
        k   = self.railproperties.k
        A   = self.railproperties.A
        G   = self.railproperties.G
        I   = self.railproperties.I
        omega_0 = np.sqrt(k*A*G/(rho*I)) # cut off frequency
        # Frequencies in Rad/s
        freq_0 = 10    # Hz 
        freq_1 = 50000 # Hz
        step   = 60   # Hz
        # Get the n first natural frequencies of beam vibration
        x_list = self.subdividespace(self.DetLambda,freq_0*np.pi*2,freq_1*np.pi*2,step*np.pi*2)
        z_list = []
        for i in range(len(x_list) - 1):
            nat_freq = scipy.optimize.brentq(self.DetLambda,x_list[i],x_list[i + 1])
            if abs(nat_freq-omega_0)>50:
                z_list.append(nat_freq)
        return sorted(z_list)[:n]
    def W(self,xi,P,omega):
        P1,P2,P3,P4 = P
        rho = self.railproperties.rho
        k   = self.railproperties.k
        A   = self.railproperties.A
        G   = self.railproperties.G
        I   = self.railproperties.I
        E   = self.railproperties.E
        L   = self.L
        omega_0 = np.sqrt(k*A*G/(rho*I)) # cut off frequency
       
        # p = omega**2*rho/(k*G)+k*A*G/(E*I)
        d = omega**2*rho*I*(1+E/(G*k))/(E*I)
        e = omega**2*(omega**2*rho**2*I/(G*k)-rho*A)/(E*I)
        delta = d**2-4*e
        x = xi*L
       
        if omega<omega_0:  
            lambda_1 = np.sqrt((d + np.sqrt(delta))/2)
            lambda_2 = np.sqrt((-d + np.sqrt(delta))/2) 
            W = P1*np.cos(lambda_1*x) + P2*np.sin(lambda_1*x) + P3*np.cosh(lambda_2*x) + P4*np.sinh(lambda_2*x )
        elif omega > omega_0:
            lambda_1 = np.sqrt((d + np.sqrt(delta))/2)
            lambda_2 = np.sqrt((d - np.sqrt(delta))/2)
            W = P1*np.cos(lambda_1*x) + P2*np.sin(lambda_1*x) + P3*np.cos(lambda_2*x) + P4*np.sin(lambda_2*x )
        return W
    def T(self,xi,P,omega):
        rho = self.railproperties.rho
        k   = self.railproperties.k
        A   = self.railproperties.A
        G   = self.railproperties.G
        I   = self.railproperties.I
        E   = self.railproperties.E
        L   = self.L
        omega_0 = np.sqrt(k*A*G/(rho*I)) # cut off frequency

        x = xi*L
        if omega<omega_0:  
            P1,P2,P3,P4 = P
            T = (((rho*omega**2)/(G*k) + (A*G*k)/(E*I))*(P2*np.cos(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2) + P4*np.cosh(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2) - P1*np.sin(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2) + P3*np.sinh(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2)) - P2*np.cos(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(3/2) + P4*np.cosh(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(3/2) + P1*np.sin(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(3/2) + P3*np.sinh(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 - (omega**2*rho*(E/(G*k) + 1))/(2*E))**(3/2))/((k*A*G)/(E*I) - (omega**2*rho)/E)
        elif omega > omega_0:
            Q1,Q2,Q3,Q4 = P
            T = (((rho*omega**2)/(G*k) + (A*G*k)/(E*I))*(Q2*np.cos(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2) + Q4*np.cos(x*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(1/2))*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(1/2) - Q1*np.sin(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2) - Q3*np.sin(x*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(1/2))*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(1/2)) - Q2*np.cos(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(3/2) - Q4*np.cos(x*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(1/2))*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(3/2) + Q1*np.sin(x*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(1/2))*(((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2 + (omega**2*rho*(E/(G*k) + 1))/(2*E))**(3/2) + Q3*np.sin(x*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(1/2))*((omega**2*rho*(E/(G*k) + 1))/(2*E) - ((omega**4*rho**2*(E/(G*k) + 1)**2)/E**2 + (4*omega**2*(A*rho - (I*omega**2*rho**2)/(G*k)))/(E*I))**(1/2)/2)**(3/2))/((k*A*G)/E*I - (omega**2*rho)/E)   
        return T
    
    
    def F_res(self,xi):
        omega_l = self.modal_nat_freqs
        F_Mod = 0
        for i in range(self.modal_n_modes):
            omega = omega_l[i]
            P = self.modal_P_coeffs[i]
            F_Mod = F_Mod + self.W(xi,P,omega)**2/omega**2

        if IsIterable(xi):
            F_res=[]
            for i in xi:
                displ_mid = self.point_load_flexibility_TIM4(i)
                F_res.append(displ_mid)
        else:
            F_res = self.point_load_flexibility_TIM4(xi)     
        F_res = np.array(F_res) - F_Mod  
        return F_res
    def M_cross_dyn(self):
        # M_ij^cross where i is i-th mode of vibration and j is j-th global dof
        rho = self.railproperties.rho
        A   = self.railproperties.A
        I   = self.railproperties.I
        
        omega_l = self.modal_nat_freqs
        M_i_cross = np.zeros((self.modal_n_modes,4))
        for i in range(self.modal_n_modes):
            omega = omega_l[i]
            P = self.modal_P_coeffs[i]
            M_i1 = self.L*integrate.quad(lambda x: rho*A*self.W(x,P,omega)*self.w(x,0) +rho*I*self.T(x,P,omega)*self.theta(x,0) ,0,1)[0]
            M_i2 = self.L*integrate.quad(lambda x: rho*A*self.W(x,P,omega)*self.w(x,1) +rho*I*self.T(x,P,omega)*self.theta(x,1) ,0,1)[0]
            M_i3 = self.L*integrate.quad(lambda x: rho*A*self.W(x,P,omega)*self.w(x,2) +rho*I*self.T(x,P,omega)*self.theta(x,2) ,0,1)[0]
            M_i4 = self.L*integrate.quad(lambda x: rho*A*self.W(x,P,omega)*self.w(x,3) +rho*I*self.T(x,P,omega)*self.theta(x,3) ,0,1)[0]
            #M_i5 = self.L*integrate.quad(lambda x: rho*A*self.W(x,P,omega)*self.w(x,4) +rho*I*self.T(x,P,omega)*self.theta(x,4) ,0,1)[0]
            M_i_cross[i,:] = [M_i1,M_i2,M_i3,M_i4]#,M_i5]
        return M_i_cross
    def M_loc_dyn(self):
        return np.eye(self.modal_n_modes)
    def K_loc_dyn(self):
        omega_l = self.modal_nat_freqs
        return np.diag(np.array(omega_l)**2)
    def C_loc_dyn(self):
        omega_l = np.array(self.modal_nat_freqs)
        alpha = 5000
        beta  = 0.0000004
        zeta = alpha/(2*omega_l)+beta*omega_l/2
        return np.diag(2*np.multiply(omega_l,zeta))
    def Psi_Modal(self,xi):
        Psi_modal =[]
        for i in range(self.modal_n_modes):
            omega = self.modal_nat_freqs[i]
            P     = self.modal_P_coeffs[i]
            Psi_modal.append(self.W(xi,P,omega))
        return Psi_modal