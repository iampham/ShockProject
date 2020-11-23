import numpy as np 
from assembleRRKK import *
from calculatePlastic import *

# Reference Mesh
n_node = 8
n_elem = 5
node_X = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],
                   [0.27,0.25],[0.75,0.27],[0.73,0.75],[0.25,0.73]])
elements = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[4,5,6,7]])

# Apply the deformation to all the boundary nodes in the mesh, for the rest just keep original coords
node_x = np.zeros(node_X.shape)
for i in range(n_node):
    X = node_X[i]
    # first initialize with the same as original
    node_x[i] = X
    # but then apply boundary conditions
    if X[0]<0.00001: 
        node_x[i,0] = 0.
    if X[1]<0.00001: 
        node_x[i,1] = 0.
    if X[0]>0.9999: 
        node_x[i,0] = 1.1
    if X[1]>0.9999: 
        node_x[i,1] = 1.


# 2 Dimension Quadrilateral Shape Function
def Nvec(xi,eta):
    return 0.25*np.array([(1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)])

def dNvecdxi(xi,eta):
    return 0.25*np.array([[(-1)*(1-eta),(+1)*(1-eta),(+1)*(1+eta),(-1)*(1+eta)],\
                          [(1-xi)*(-1),(1+xi)*(-1),(1+xi)*(+1),(1-xi)*(+1)]])

# Parameters we use in function
Gamma = 0.7 # Mie Gruneisen Parameter []
T = 303.15 # Ambient temperature of material [K]
T_0 = 300 # Reference temperature of material [K]
v = 0.51e3 # (ARBITRARY, something less that v_0) Specific volume of material at temp T [m^3/Kg]
rho_0 = 1.891e3 # Initial density [Kg/m^3] 
v_0 = 1/rho_0 # specific volume of material at reference temp [m^3/Kg]
K_0 = 17.822e9 # Reference bulk modulus [Pa]
C_v = 2357.3 # Specific heat capacity [J/(Kg*K)]
s = 1.79 # Slope Hugoniot []
alpha = np.array([[1,0],[0,1]]) # Thermal expansion tensor []
gamma_dot_ref = 0.001e-9 # Reference slip rate [s]
m = 0.1 # Slip rate exponent []
g = 1
g_sat = 155.73e6 # Saturation slip resistance [Pa]
g_prev = 1*np.ones([10,1]) # NEED CORRECT VALUE [Pa]
a = 2.5 # Hardening exponent []
h = 9.34e6 # Hardening matrix [Pa]
# Elastic Tensor CC
C_ela = 1e9*np.array([[21.15,10.18,9.77,0.058,4.05,-0.18],
                      [10.18,20.34,13.35,0.23,6.96,0.14],
                      [9.77,13.35,21.27,-0.004,5.01,0.19],
                      [0.058,0.23,-0.004,8.79,0.32,4.16],
                      [4.05,6.96,5.01,0.32,6.20,0.22],
                      [-0.18,0.14,0.19,4.16,0.22,10.00]])


const_dictionary={"Gamma" : 0.7, # Mie Gruneisen Parameter []
"T" : 303.15, # Ambient temperature of material [K]
"T_0" : 300, # Reference temperature of material [K]
"v" : 0.51e3, # (ARBITRARY, something less that v_0) Specific volume of material at temp T [m^3/Kg]
"rho_0" : 1.891e3 ,# Initial density [Kg/m^3] 
"v_0" : 1/1.891e3, # specific volume of material at reference temp [m^3/Kg]
"K_0" : 17.822e9, # Reference bulk modulus [Pa]
"C_v" : 2357.3, # Specific heat capacity [J/(Kg*K)]
"s" : 1.79, # Slope Hugoniot []
"alpha" : np.array([[1,0],[0,1]]), # Thermal expansion tensor []
"gamma_dot_ref" : 0.001e-9, # Reference slip rate [s]
"m" : 0.1, # Slip rate exponent []
"g" : 1, #
"g_sat" : 155.73e6, # Saturation slip resistance [Pa]
"g_prev" : 1*np.ones([10,1]), # NEED CORRECT VALUE [Pa]
"a" : 2.5, # Hardening exponent []
"h" : 9.34e6 # Hardening matrix [Pa]
}
assembleRRKK(const_dictionary, Nvec, dNvecdxi, n_node, n_elem, elements, node_X, node_x)