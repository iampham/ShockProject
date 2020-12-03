from computeSecondPiola import *
from calculatePlastic import *
import numpy as np
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
Gamma = const_dictionary["Gamma"] # Mie Gruneisen Parameter []
T = const_dictionary["T"] # Ambient temperature of material [K]
T_0 = const_dictionary["T_0"] # Reference temperature of material [K]
rho_0 = const_dictionary["rho_0"] # Initial density [Kg/m^3] 
v_0 = const_dictionary["v_0"] # specific volume of material at reference temp [m^3/Kg]
K_0 = const_dictionary["K_0"] # Reference bulk modulus [Pa]
C_v = const_dictionary["C_v"] # Specific heat capacity [J/(Kg*K)]
s = const_dictionary["s"] # Slope Hugoniot []
alpha = const_dictionary["alpha"] # Thermal expansion tensor []
gamma_dot_ref = const_dictionary["gamma_dot_ref"] # Reference slip rate [s]
m = const_dictionary["m"] # Slip rate exponent []
g_sat = const_dictionary["g_sat"]# Saturation slip resistance [Pa]
a = const_dictionary["a"] # Hardening exponent []
h = const_dictionary["h"]# Hardening matrix [Pa]
C_ela_2d_voigt = const_dictionary["C_ela_2d_voigt"]
C_ela_3d=const_dictionary["C_ela_3d"]
v=v_0
S_prev=np.ones([2,2])

# ANDREW's DEBUGGING SESSION 1: NEED A TINY DEFORMATION GRADIENT, STRESSES ARE GIANT
F_p_prev_loc=np.eye(2)
F_e_prev_loc=np.eye(2)*.999
F= np.dot(F_e_prev_loc,F_p_prev_loc)
g_prev_loc=np.ones([10,1])*100e6
dt=1e-10
g_prev = np.ones([10,1,8,4])*100e6
ei=0
ip=0

# Slip resistance stop parameter
g_tol = 1e-4
# Initialize g_max
g_max = 1
g_itermax = 100
g_iter = 0
x = np.zeros((2))

g_diff = 1

# ANDREW DEBUGGING: Created the g_not_converged flag, Moved g_prev_loc definition from the body of the while
# loop into the initialization step. 
g_not_converged = True
g_prev_loc=g_prev[:,:,ei,ip]


while (g_not_converged and g_iter<g_itermax):
    
    # Pre solve stress
    # according to the current shockwave boundary and coordinates of ip, update v
    if 1:
        S_prev,S_eos,S_el_voigt,p_eos=computeSecondPiola(F_e_prev_loc,const_dictionary,v)
    else:
        S_prev,S_eos,S_el_voigt,p_eos=computeSecondPiola(F_e_prev_loc,const_dictionary,v_0)

    # Initialize stress residual
    res_S = 1.
    norm_res_S = 1
    S_iter = 0
    S_tol = 1e-5
    S_itermax = 1000

    # Solve Stress newton raphson
    while (norm_res_S>S_tol and S_iter<S_itermax):

    
        # according to the current shockwave boundary and coordinates of ip, update v
        
        # Residual and Jacobian to compute Next Stress                    
        res_S, J_S, F_e_current = computeSecondPiolaResidualJacobian(S_prev,F_p_prev_loc,F,g_prev_loc,dt,const_dictionary) 
        #print(J_S)
        # compute delta_S and add it to S
        #delta_S = -np.tensordot(np.linalg.inv(J_S),res_S,axes=2)
        J_S=J_S[0:2,0:2,0:2,0:2]
        res_S=res_S[0:2,0:2]
        delta_S = -np.tensordot(np.linalg.tensorinv(J_S),res_S,axes=2)
        # print(delta_S)
        S_next = S_prev + delta_S[0:2,0:2]
        S_next = S_next[0:2,0:2] # Go back to 2D
        # Stop criteria
        norm_res_S = np.linalg.norm(res_S)

        # # TODO Need to add shock condition to next guess of S_prev
        # if x[0]<shock_bound:
        #     S_prev,S_eos,S_el_voigt,p_eos=computeSecondPiola(F_e_prev_loc,const_dictionary,v)
        # else:
        #     S_prev,S_eos,S_el_voigt,p_eos=computeSecondPiola(F_e_prev_loc,const_dictionary,v_0)
        # # Iteration to calculate actual split of F=F_p*F_e
        S_prev=S_next
        S_iter += 1
        # print("S_iter: ",S_iter)
        # print("res:",norm_res_S)

    print("res:", norm_res_S)
    #print("S", S_prev)
    # g_sub
    F_e_current = F_e_current[0:2,0:2]

    F_p_next, g_current = calculateNextPlastic(F_p_prev_loc,gamma_dot_ref, m, g_sat, g_prev_loc, a, h, dt, F_e_current, S_next)
    
    print("g_current",g_current)
    # g_diff = np.abs(g_prev_loc-g_current)

    # g_diff = np.linalg.norm(g_diff)
    # if g_diff > g_tol:
    #     g_max = g_diff

    g_not_converged = notConvergedYet(g_prev_loc,g_current,g_tol)
    g_iter += 1
    g_prev_loc=g_current
    print("g_iter: ",g_iter)

    # print(np.linalg.det(F_p_next))

    if np.linalg.norm(g_diff) == np.nan:
        break
