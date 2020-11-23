
import numpy as np 
from assembleRRKK import *
from calculatePlastic import *


# Loop over time steps n 
timeStart = 0
timeEnd = 100
nSteps = 100
t_vec = np.linspace(timeStart,timeEnd, nSteps)

# Calculate the size of the timestep
deltat = timeVec[1] - timeVec[0]


# Initialize mesh
n_nodes = 8
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

# Elastic Tensor CC
C_ela = 1e9*np.array([[21.15,10.18,9.77,0.058,4.05,-0.18],\
                      [10.18,20.34,13.35,0.23,6.96,0.14],\
                      [9.77,13.35,21.27,-0.004,5.01,0.19],\
                      [0.058,0.23,-0.004,8.79,0.32,4.16],\
                      [4.05,6.96,5.01,0.32,6.20,0.22],\
                      [-0.18,0.14,0.19,4.16,0.22,10.00]]) # elasticity tensor for a 3D problem
C_ela = C_ela[0:3,0:3] # Elasticity tensor for a 2D problem

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
"a" : 2.5, # Hardening exponent []
"h" : 9.34e6, # Hardening matrix [Pa]
"C_ela" : C_ela
}




# Intialize internal variables
S_all = np.zeros([2,2,n_elem,nsteps]) # blah
T_all = const_dictionary["T"] * np.ones(nSteps) # Kelvin, should be nnodes* ntime?
p_all = np.zeros([nsteps])

F_all = np.zeros([2,2,n_elem,nSteps])
F_e_all = np.zeros([2,2,n_elem,nSteps])
F_p_all = np.zeros([2,2,n_elem,nSteps])
g_all = np.zeros([10,1,n_elem,nSteps])

# Initialize first guess for plastic deformation gradient
F_p_all[:,:,0] = np.eye(2)


# Keep track of displacements at all time steps
u_vec = np.zeros((n_nodes,2,nSteps))
v_vec = np.zeros((n_nodes,2,nSteps))
a_vec = np.zeros((n_nodes,2,nSteps))



# Initial guess of 0 displacement
u_current = np.zeros(n_nodes,2)
v_current = np.zeros(n_nodes,2)
g_prev=g_all[:,:,:,0]

# TODO: define M, the mass matrix. 
M = 

# Need to calculate acceleration at current timestep
RR, KK,F_p_next,g_next = assembleRRKK(const_dictionary,Nvec, dNvecdxi, n_node, n_elem, elements, node_X, node_x,F_p_prev,g_prev,deltat)
a_current = np.dot(Minv, -np.dot(C,v_current) - RR + P_current)


# Initialize external loading vector
P_vec = np.zeros()



for tIndex in range(1, len(t_vec)):

       
    # Update velocity, displacement
    v_next = v_current +1/2 * deltat (a_current + a_next)
    u_next = u_current + deltat*v_current + (deltat)**2/2 *a_current
    
    # Newton raphson for global problem
    res = 1
    iter = 0
    tol = 1e-5
    itermax = 10
    while(res>tol and iter<itermax):

        RR, KK,F_p_next,g_next= assembleRRKK(const_dictionary,Nvec, dNvecdxi, n_node, n_elem, elements, node_X, node_x,F_p_prev,deltat)

        RRdof= RR[8:]
        KKdof = KK[8:, 8:]

        res = np.linalg.norm(RRdof)
        incr_u = -np.linalg.solve(KKdof,RRdof)
        iter +=1 

        for i in range(4):
            node_x[4+i,0] += incr_u[i*2]
            node_x[4+i,1] += incr_u[i*2+1]
        iter +=1
        print('iter %i'%iter)
        print(res)

    # Update acceleration
    MassDampInv = np.linalg.inv(M + deltat/2 * C)
    a_next = np.dot(MassDampInv, ( P_next - RR - deltat/2 * np.dot(C,a_current)) )


    # Assign variables to the vector so they don't get lost
    u_vec[:,:,tIndex] = u_current
    v_vec[:,:,tIndex] = v_current
    a_vec[:,:,tIndex] = a_current

    # 
    u_current = u_next
    v_current = v_next
    a_current = a_next