
import numpy as np 
from assembleRRKK import *
from calculatePlastic import *
from mat2tens import *
import time

# Parameters we use in function

# Elastic Tensor CC # TODO verify 
# C_ela_3d_voigt = 1e9*np.array([[21.15,10.18,9.77,0.058,4.05,-0.18],\
#                       [10.18,20.34,13.35,0.23,6.96,0.14],\
#                       [9.77,13.35,21.27,-0.004,5.01,0.19],\
#                       [0.058,0.23,-0.004,8.79,0.32,4.16],\
#                       [4.05,6.96,5.01,0.32,6.20,0.22],\
#                       [-0.18,0.14,0.19,4.16,0.22,10.00]]) # elasticity tensor for a 3D problem
# C_ela_2d_voigt = C_ela_3d_voigt[0:3,0:3] # Elasticity tensor for a 2D problem

# Isotropic tensor
# material parameters of beta-HMX from 
# Effect of initial damage variability on hot-spot nucleation in energetic materials
# Camilo A. Duarte, Nicolò Grilli, and Marisol Koslowski
E= 25.125e4# should be 25.12e9
nu= 0.24
C_ela_3d_voigt =np.zeros([6,6])
C_ela_3d_voigt[0,0]=E*(1-nu)/((1+nu)*(1-2*nu))
C_ela_3d_voigt[1,1]=E*(1-nu)/((1+nu)*(1-2*nu))
C_ela_3d_voigt[2,2]=E*(1-nu)/((1+nu)*(1-2*nu))
C_ela_3d_voigt[3,3]=E/(2*(1+nu))
C_ela_3d_voigt[4,4]=E/(2*(1+nu))
C_ela_3d_voigt[5,5]=E/(2*(1+nu))
C_ela_3d_voigt[0,1]=E*nu/((1+nu)*(1-2*nu))
C_ela_3d_voigt[0,2]=E*nu/((1+nu)*(1-2*nu))
C_ela_3d_voigt[1,2]=E*nu/((1+nu)*(1-2*nu))
C_ela_3d_voigt[1,0]=E*nu/((1+nu)*(1-2*nu))
C_ela_3d_voigt[2,0]=E*nu/((1+nu)*(1-2*nu))
C_ela_3d_voigt[2,1]=E*nu/((1+nu)*(1-2*nu))



C_ela_3d = mat2tens(C_ela_3d_voigt) #  3 3 3 3
C_ela_2d = C_ela_3d[0:2,0:2,0:2,0:2]
C_ela_2d_voigt=np.zeros([3,3])
C_ela_2d_voigt[0:2,0:2] = C_ela_3d_voigt[0:2,0:2] 
C_ela_2d_voigt[2,2] = C_ela_3d_voigt[3,3]


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
"g_sat" : 155.73e3, # Saturation slip resistance [Pa] was e6 # TODO 7 dec 2020 DM - verified this covnerged with 155.73e6 and NR tol 1e-5
"a" : 2.5, # Hardening exponent []
"h" : 9.34e6, # Hardening matrix [Pa]
"C_ela_2d_voigt" : C_ela_2d_voigt,
"C_ela_3d": C_ela_3d, #  3 3 3 3
"C_ela_2d": C_ela_2d,
"C_s": 3070., # reference bulk speed of sound [m/s] 
"n_IP":4 # integration pts per elem
}

# Loop over time steps n 
# TODO: start time changed here
timeStart = 0.
timeEnd =timeStart+ 0.5
nSteps = 30
t_vec = np.linspace(timeStart,timeEnd, nSteps)

# Calculate the size of the timestep
deltat = t_vec[1] - t_vec[0]

# particle vel
U_p=2000 # 100m/s
# assuming that the material does not undergo phase transition, Us is linear with Up
U_s=const_dictionary["C_s"]+const_dictionary["s"]*U_p
# U_s is made faster to increase the simulation speed


# find v1 
const_dictionary["rho_0"]=1/const_dictionary["v_0"]
rho_1=const_dictionary["rho_0"]*U_s/(U_s-U_p)
v1=1/rho_1
const_dictionary["v"]=v1 

# Initialize mesh
n_nodes = 8
n_elem = 5
node_X = np.array([[0.,0.],[1.,0.],[1.,1.],[0.,1.],
                   [0.27,0.25],[0.75,0.27],[0.73,0.75],[0.25,0.73]])
elements = np.array([[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7],[4,5,6,7]])

# Apply the deformation to all the boundary nodes in the mesh, for the rest just keep original coords
node_x = np.zeros(node_X.shape)
for i in range(n_nodes):
    X = node_X[i]
    # first initialize with the same as original
    node_x[i] = X
    # but then apply boundary conditions
    if X[0]<= 0.00001:
        node_x[i,0] = 0.01 #known displacement, no more shocks
    if X[1]<0.00001: 
        node_x[i,1] = 0.
    if X[0]>0.9999: 
        node_x[i,0] = 1.
    if X[1]>0.9999: 
        node_x[i,1] = 1.




# 2 Dimension Quadrilateral Shape Function
def Nvec(xi,eta):
    return 0.25*np.array([(1-xi)*(1-eta),(1+xi)*(1-eta),(1+xi)*(1+eta),(1-xi)*(1+eta)])

def dNvecdxi(xi,eta):
    return 0.25*np.array([[(-1)*(1-eta),(+1)*(1-eta),(+1)*(1+eta),(-1)*(1+eta)],\
                          [(1-xi)*(-1),(1+xi)*(-1),(1+xi)*(+1),(1-xi)*(+1)]])







# Intialize internal variables
n_IP = 4 # Integration points
S_all = np.zeros([2,2,n_elem,n_IP,nSteps]) # Contains Stress S for each element and each integration point
sigma_all = np.zeros([2,2,n_elem,n_IP,nSteps])
T_all = const_dictionary["T"] * np.ones(nSteps) # TODO (Depends on what temp we put in dicitionary) Kelvin, should be nnodes* ntime?
# p_all = np.zeros([nSteps])

F_all = np.zeros([2,2,n_elem,n_IP,nSteps])
F_e_all = np.zeros([2,2,n_elem,n_IP,nSteps])
F_p_all = np.zeros([2,2,n_elem,n_IP,nSteps])
g_all = np.zeros([10,1,n_elem,n_IP,nSteps])

# Intitial condition of hardening
g_all[:,:,:,:,0] =103.03e3 * np.ones([10,1,n_elem,n_IP]) # 1e7 times smaller than g_dot_ref? # TODO 7 dec 2020 DM - In process of testing to same order as g_sat from 103.03e3 to 103.03e6

# Initialize first guess for plastic deformation gradient
for i_elem in range(n_elem):
    for i_p in range(n_IP): # 4 
        F_p_all[:,:,i_elem,i_p,0] = np.eye(2)

# Keep track of displacements at all time steps
u_vec = np.zeros([n_nodes,2,nSteps])
# v_vec = np.zeros((n_nodes,2,nSteps))
# a_vec = np.zeros((n_nodes,2,nSteps))

# Initial guess of 0 displacement
# u_current = np.zeros(n_nodes,2)
# v_current = np.zeros(n_nodes,2)
g_prev=g_all[:,:,:,:,0]
F_p_prev = F_p_all[:,:,:,:,0]
S_prev = S_all[:,:,:,:,0]
sigma_prev = sigma_all[:,:,:,:,0]
resultant_increment_prev = np.zeros([2,2])


# # TODO: define M, the mass matrix. 
# M = 
# Minv = 

# Need to calculate acceleration at current timestep
# Newton raphson for global problem
res = 1.
iter = 0
tol = 1e-3
itermax = 1000



print("Time Step 0")

# newton raphson for initial time step
while(res>tol and iter<itermax):

    RR, KK,F_p_next,g_next,S_next,F_e_next,F_next,sigma_next= assembleRRKK(const_dictionary,Nvec, dNvecdxi, n_nodes, n_elem, elements, node_X, node_x,F_p_prev,g_prev,deltat)

    RRdof= RR[8:]
    KKdof = KK[8:, 8:]
    # a relatively large residual for ei3 ip2, a big increment occured after first iteration here, incru=-130 for one node
    res = np.linalg.norm(RRdof)
    # print("KKdof",KKdof)
    # print("RRdof",RRdof)
    # print("KKdof_inv",np.linalg.inv(KKdof))
    incr_u = -np.linalg.solve(KKdof,RRdof)

    # a_current = np.dot(Minv, -np.dot(C,v_current) - RR ) # TODO: check dimensions of matrices, P_current set to 0
    # const_dictionary["C_ela"]

    # iter +=1 

    for i in range(4):
        node_x[4+i,0] += incr_u[i*2]
        node_x[4+i,1] += incr_u[i*2+1]
    iter +=1
    print("res:",res)
    print("incr_u",incr_u)
    print("Fpnext", F_p_next[:,:,4,0])
    # print('KK', np.linalg.norm(KKdof))
    # print('RR', np.linalg.norm(RRdof))

print('NR iterations %i, res %1.7e'%(iter,res))


# store in timed var
# u_vec[:,:,0]=incr_u
u_vec[:,:,0] = node_x # Storing deformed x instead of displacement u
# v_vec[:,:,0]=v_current
# a_vec[:,:,0]=a_current
S_all[:,:,:,:,0] = S_next
F_all[:,:,:,:,0] = F_e_next
F_e_all[:,:,:,:,0] = F_next
F_p_all[:,:,:,:,0] = F_p_next
g_all[:,:,:,:,0] = g_next
sigma_all[:,:,:,:,0] = sigma_next


# Initialize external loading vector
# P_vec = np.zeros()


# start of time loop

for tIndex in range(1, len(t_vec)):

    print("\nTime Step",tIndex)

    # update boudary conditions
    # Apply the deformation to all the boundary nodes in the mesh, for the rest just keep original coords
    for i in range(n_nodes):
        X = node_X[i]
        # but then apply boundary conditions
        if X[0]<0.00001:
            node_x[i,0] += 0.3 * deltat # TODO: make a time dependent BC
        if X[1]<0.00001: 
            node_x[i,1] = 0.
        # right boundary fixed for all time
        if X[0]>0.9999: 
            node_x[i,0] = 1.
        if X[1]>0.9999: 
            node_x[i,1] = 1.

    # Update velocity, displacement
    # v_next = v_current +1/2 * deltat (a_current + a_next)
    # u_next = u_current + deltat*v_current + (deltat)**2/2 *a_current
    
    # Update prev variables for this time step
    g_prev=g_all[:,:,:,:,tIndex-1]
    F_p_prev = F_p_all[:,:,:,:,tIndex-1]

    # Newton raphson for global problem
    res = 1
    iter = 0
    while(res>tol and iter<itermax):

        RR, KK,F_p_next,g_next,S_next,F_e_next,F_next,sigma_next = assembleRRKK(const_dictionary,Nvec, dNvecdxi, n_nodes, n_elem, elements, node_X, node_x,F_p_prev,g_prev,deltat)

        RRdof= RR[8:]
        KKdof = KK[8:, 8:]

        res = np.linalg.norm(RRdof)
        incr_u = -np.linalg.solve(KKdof,RRdof)
        
        for i in range(4):
            node_x[4+i,0] += incr_u[i*2]
            node_x[4+i,1] += incr_u[i*2+1]
        iter +=1
        print("res:",res)
        print("incr_u",incr_u)
        

    print('NR iterations %i, res %1.7e'%(iter,res))

    # Update acceleration
    # MassDampInv = np.linalg.inv(M + deltat/2 * C)
    # a_next = np.dot(MassDampInv, (  RR - deltat/2 * np.dot(C,a_current)) )
    
    # a_current = np.dot(Minv, -np.dot(C,v_current) - RR ) # TODO: check dimensions of matrices, P_current set to 0
    # store in timed var
    # u_vec[:,:,tIndex]=incr_u
    u_vec[:,:,tIndex] = node_x # Storing deformed x instead of displacement u
    # v_vec[:,:,tIndex]=v_current
    # a_vec[:,:,tIndex]=a_current
    S_all[:,:,:,:,tIndex] = S_next
    F_all[:,:,:,:,tIndex] = F_next
    F_e_all[:,:,:,:,tIndex] = F_e_next
    F_p_all[:,:,:,:,tIndex] = F_p_next
    sigma_all[:,:,:,:,tIndex] = sigma_next

    g_all[:,:,:,:,tIndex] = g_next
    # print("F_next",F_next[:,:,4,0])
    print("F_e_next",F_e_next[:,:,4,0])
    print("F_p_next",F_p_next[:,:,4,0])
    # print("S_next",S_next[:,:,4,0])

    F_p_prev=F_p_next
    g_prev=g_next
    # 
    # u_current = u_next
    # v_current = v_next
    # a_current = a_next
    filename = 'FEAData' 
    np.savez(filename, S_all = S_all, sigma_all = sigma_all, F_all = F_all, F_e_all = F_e_all, F_p_all = F_p_all, g_all = g_all,timeStart=timeStart,timeEnd=timeEnd,nSteps=nSteps, n_IP=n_IP, n_elem=n_elem, u_vec = u_vec, node_X = node_X)



# Saves files in .npz file. To read, just use the command np.load('FEAData.npz'). 
# The load command will output a dictionary containing all of the data shown here. 
# To access the data, access it like you would a dictionary. 
filename = 'FEAData' 
np.savez(filename, S_all = S_all, sigma_all = sigma_all, F_all = F_all, F_e_all = F_e_all, F_p_all = F_p_all, g_all = g_all,timeStart=timeStart,timeEnd=timeEnd,nSteps=nSteps, n_IP=n_IP, n_elem=n_elem, u_vec = u_vec, node_X = node_X)