# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:21:46 2020

Vehicle Assembly, default parameters based on an EC EW4 Panorama waggon      

@author: Cyprien Hoelzl
"""
import numpy as np
      
class VehicleAssembly():
    # Assembly of vehicle
    def __init__(self,axle = None,bogie = None, carbody = None):
        # custom axles, bogies and bodies can be passed as inputs
        if axle is None:
            axle = Axle()
        if bogie is None:
            bogie = Bogie()
        if carbody is None:
            carbody = CarBody()
        self.Axle = axle
        self.Bogie = bogie
        self.CarBody = carbody
        self.assembleVehicleMatrices2()
    def assembleVehicleMatrices(self):
        """
        Quarter car vehicle assembly when including only car and axle parameters
         - Assemble the M, K, C matrices
         - Save the DOF names
        #  ________
        # |________|   body
        #     z        primary suspension
        #     O        axle

        Returns
        -------
        None.

        """
        self.M_veh = np.eye(2)*[self.CarBody.M_car, self.Axle.M_axle]
        self.K_veh = np.array([[ self.Axle.K_prim_spring,- self.Axle.K_prim_spring],
                               [-self.Axle.K_prim_spring, self.Axle.K_prim_spring]])
        self.C_veh = np.array([[ self.Axle.D_prim_damper,- self.Axle.D_prim_damper],
                               [-self.Axle.D_prim_damper, self.Axle.D_prim_damper]])
        self.u_names = ['w_body_I','w_bogie_1','w_axle_1']
              
    def assembleVehicleMatrices2(self):
        """
        Quarter car vehicle assembly when including body, bogie and axle
         - Assemble the M, K, C matrices
         - Save the DOF names
        #  ________
        # |________|   body
        #     z        secondary suspension
        #   |___|      bogie
        #     z        primary suspension
        #     O        axle

        Returns
        -------
        None.

        """
        self.M_veh = np.eye(3)*[self.CarBody.M_car, self.Bogie.M_bogie, self.Axle.M_axle]
        self.K_veh = np.array([[self.Bogie.K_sec_susp,   - self.Bogie.K_sec_susp,       0],
                               [- self.Bogie.K_sec_susp, self.Axle.K_prim_spring+self.Bogie.K_sec_susp,  - self.Axle.K_prim_spring],
                               [0,                       -self.Axle.K_prim_spring,                    self.Axle.K_prim_spring]])
        self.C_veh = np.array([[self.Bogie.D_sec_susp,   - self.Bogie.D_sec_susp,       0],
                               [- self.Bogie.D_sec_susp,  self.Bogie.D_sec_susp+self.Axle.D_prim_damper,- self.Axle.D_prim_damper],
                               [0, -self.Axle.D_prim_damper, self.Axle.D_prim_damper]])
        self.u_names = ['w_body_I','w_bogie_1','w_axle_1']
class Axle():
    def __init__(self, m_axle=(1.1843E+3)/2, k_prim_spring = 1.200E6, d_prim_damper=0):
        # Initialize Axle Parameters
        self.M_axle = m_axle # [kg]
        self.K_prim_spring = k_prim_spring #[N/m]
        self.D_prim_damper = d_prim_damper #2.7333E+3 # [Ns/m]
class CarBody():
    def __init__(self,m_car=37000/8):
        # Initialize Car Body Parameters
        self.M_car = m_car # [kg]
class Bogie():
    def __init__(self, m_bogie=(3000+784+200)/4, k_sec_susp=491.000E+3/4, d_sec_susp=2.650E+6/2):
        # Initialize Bogie Parameters
        self.M_bogie = m_bogie  # [kg]
        self.K_sec_susp = k_sec_susp# [N/m]
        self.D_sec_susp = d_sec_susp# [Ns/m]
if __name__ == "__main__":        
    Vehicle = VehicleAssembly()    
    print('M_veh =\n',Vehicle.M_veh,
          '\n','K_veh = \n',Vehicle.K_veh,'\n',
          'C_veh = \n',Vehicle.C_veh)
