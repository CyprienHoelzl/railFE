# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:21:46 2020

Vehicle Assembly      

@author: CyprienHoelzl
"""
import numpy as np
      
class VehicleAssembly():
    def __init__(self):
        self.Axle = Axle()
        self.Bogie = Bogie()
        self.CarBody = CarBody()
        self.assembleVehicleMatrices2()
    def assembleVehicleMatrices(self):
        self.M_veh = np.eye(2)*[self.CarBody.M_car, self.Axle.M_axle]
        self.K_veh = np.array([[ self.Axle.K_prim_spring,- self.Axle.K_prim_spring],
                               [-self.Axle.K_prim_spring, self.Axle.K_prim_spring]])
        self.C_veh = np.array([[ self.Axle.D_prim_damper,- self.Axle.D_prim_damper],
                               [-self.Axle.D_prim_damper, self.Axle.D_prim_damper]])
        self.u_names = ['w_body_I','w_bogie_1','w_axle_1']
        
        
    def assembleVehicleMatrices2(self):
        self.M_veh = np.eye(3)*[self.CarBody.M_car, self.Bogie.M_bogie, self.Axle.M_axle]
        self.K_veh = np.array([[self.Bogie.K_sec_susp,   - self.Bogie.K_sec_susp,       0],
                               [- self.Bogie.K_sec_susp, self.Axle.K_prim_spring+self.Bogie.K_sec_susp,  - self.Axle.K_prim_spring],
                               [0,                       -self.Axle.K_prim_spring,                    self.Axle.K_prim_spring]])
        self.C_veh = np.array([[self.Bogie.D_sec_susp,   - self.Bogie.D_sec_susp,       0],
                               [- self.Bogie.D_sec_susp,  self.Bogie.D_sec_susp+self.Axle.D_prim_damper,- self.Axle.D_prim_damper],
                               [0, -self.Axle.D_prim_damper, self.Axle.D_prim_damper]])
        self.u_names = ['w_body_I','w_bogie_1','w_axle_1']
class Axle():
    # Simple Axle with mass
    def __init__(self):
        self.M_axle = (1.1843E+3)/2  #(1.1843E+3)/2   # [kg]
        self.K_prim_spring = 1.200E6+5.043165E6#+1E8 # [N/m] 
        print('Warning: increased primary spring')
        self.D_prim_damper = 0 #2.7333E+3 # [Ns/m]
class CarBody():
    # Simple Vehicle with mass
    def __init__(self):
        self.M_car = (37000)/8   # [kg]
class Bogie():
    # Simple Vehicle with mass
    def __init__(self):
        self.M_bogie = (3000+784+200)/4#(37000)/8/4   # [kg]
        self.K_sec_susp = 491.000E+3/4
        self.D_sec_susp = 2.650E+6/2
if __name__ == "__main__":        
    Vehicle = VehicleAssembly()    
    print('M_veh =\n',Vehicle.M_veh,
          '\n','K_veh = \n',Vehicle.K_veh,'\n',
          'C_veh = \n',Vehicle.C_veh)
