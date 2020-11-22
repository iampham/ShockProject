
import numpy as np 
from assembleRRKK import *
from calculatePlastic import *
# Intialize internal vairables
S = 0 # blah


# Loop over time steps n 
timeStart = 0
timeEnd = 100
nSteps = 100
t_vec = np.linspace(timeStart,timeEnd, nSteps)

# Calculate the size of the timestep
deltat = timeVec[1] - timeVec[0]


# Initial guess of 0 displacement
u_current = np.array([0], [0])
v_current = np.array([0], [0]) 

# Need to calculate acceleration at current timestep
RR, KK = assembleRRKK(u_current)
a_current = np.dot(Minv, -np.dot(C,v_current) - RR + P_current)


# Initialize external loading vector
P_vec = np.zeros()

# Keep track of displacements at all time steps
u_vec = np.zeros((2,2,nSteps))
v_vec = np.zeros((2,2,nSteps))
a_vec = np.zeros((2,2,nSteps))

for tIndex in range(len(t_vec)):

       
    # Update velocity, displacement
    v_next = v_current +1/2 * deltat (a_current + a_next)
    u_next = u_current + deltat*v_current + (deltat)**2/2 *a_current
    
    # Newton raphson for global problem
    RR,KK = assembleRRKK(u_next)


    # Update acceleration
    MassDampInv = np.linalg.inv(M + deltat/2 * C)
    a_next = np.dot(MassDampInv, ( P_next - RR - deltat/2 * np.dot(C,a_current)) )


    # Assign variables to the vector so they don't get lost
    u_vec[tIndex] = u_current
    v_vec[tIndex] = v_current
    a_vec[tIndex] = a_current

    # 
    u_current = u_next
    v_current = v_next
    a_current = a_next