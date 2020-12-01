import numpy as np
from computeSecondPiola import *
from mat2tens import *


# Elastic Tensor CC # TODO verify 
C_ela_3d_voigt = 1e9*np.array([[21.15,10.18,9.77,0.058,4.05,-0.18],\
                      [10.18,20.34,13.35,0.23,6.96,0.14],\
                      [9.77,13.35,21.27,-0.004,5.01,0.19],\
                      [0.058,0.23,-0.004,8.79,0.32,4.16],\
                      [4.05,6.96,5.01,0.32,6.20,0.22],\
                      [-0.18,0.14,0.19,4.16,0.22,10.00]]) # elasticity tensor for a 3D problem
C_ela_2d_voigt = C_ela_3d_voigt[0:3,0:3] # Elasticity tensor for a 2D problem

C_ela_3d = mat2tens(C_ela_3d_voigt) #  3 3 3 3

const_dictionary={"Gamma" : 0.7, # Mie Gruneisen Parameter []
"T" : 300., # TODO ARBITRARY VALUE FOR NOW Ambient temperature of material [K]
"T_0" : 300., # Reference temperature of material [K]
"rho_0" : 1.891e3 ,# Initial density [Kg/m^3] 
"v_0" : 1/1.891e3, # specific volume of material at reference temp [m^3/Kg]
"K_0" : 17.822e9, # Reference bulk modulus [Pa]
"C_v" : 2357.3, # Specific heat capacity [J/(Kg*K)]
"s" : 1.79, # Slope Hugoniot []
"alpha" : np.array([[1,0],[0,1]]), # TODO NEED RIGHT TENSOR hermal expansion tensor []
"gamma_dot_ref" : 0.001e9, # Reference slip rate [s^-1]
"m" : 0.1, # Slip rate exponent []
"g_sat" : 155.73e6, # Saturation slip resistance [Pa]
"a" : 2.5, # Hardening exponent []
"h" : 9.34e6, # Hardening matrix [Pa]
"C_ela_2d_voigt" : C_ela_2d_voigt,
"C_ela_3d": C_ela_3d, #  3 3 3 3
"C_s": 3070., # reference bulk speed of sound [m/s] 
"n_IP":4 # integration pts per elem
}

S_prev=np.ones([3,3])
F_p_prev=np.eye(3)*0.5
F=np.eye(3)
g=np.ones([10,1])*1e6
dt=0.1


J_S=computeSecondPiolaJacobian(S_prev,F_p_prev,F,g,dt,const_dictionary)

print(J_S)
print(np.shape(J_S))
