import numpy as np 
import time
def calculateNextPlastic(F_p,gamma_dot_ref, m,  g_sat, g_prev, a, h, dt, F_e, S):
    
    
    result_inc, g_current=calculateResultantIncrement(gamma_dot_ref, m,  g_sat, g_prev, a, h, dt, F_e, S)
    # print("result_inc, " ,result_inc) # result inc is huge e80 for now


    F_p_current=np.dot(result_inc,F_p)

    return F_p_current,g_current

def calculateResultantIncrement(gamma_dot_ref, m,  g_sat, g_prev, a, h, dt, F_e,S):
    """ 
    Calculates the plastic part of the deformation tensor. Uses the following equation:
    F_p = deltaGamma dot F_p_prev
    deltaGamma = 
    

    INPUTS: 
    gamma_dot: reference slip rate
    m: slip rate exponent

    g - resistance vector
    g_sat
    h_0
    q_sisj
    
    # 3 adding deifitions
    a - Hardening Exponent
    h - Hardening matrix

    
    """ 

    F_e_3 = np.zeros([3,3])
    # 4 - Fixed this should 0:2, not 0:1

    F_e_3[0:2,0:2] = F_e
    F_e_3[2,2] = 1 # TODO changed to 0 for debugging


    S3 = np.zeros([3,3])
    S3[0:2,0:2] = S
    S3[2,2] = 0  # TODO changed to 0 for debugging

    strainIncrement = np.zeros([3,3])

    slipRates = np.zeros([10,1])

    for index in range(10):

        slipRates[index], schmidTensor,tau_th_s,tau_s = calculateSlipRate(F_e_3,S3,gamma_dot_ref, m, g_prev[index], index)
        # print("slipRates[index]",slipRates[index])
        # print("schmidTensor",schmidTensor)
        strainIncrement += slipRates[index] * dt* schmidTensor
    
    resultant_increment=np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i==j:
                resultant_increment[i,j] += 1 - strainIncrement[i,j]
            else:
                resultant_increment[i,j] += 0 - strainIncrement[i,j]


    g_next = getNextResistance(g_sat, g_prev, a, h, slipRates, dt)

    return resultant_increment[0:2, 0:2], g_next

def calculateSlipRate(F,S,gamma_dot_ref, m, g_si, index):
    # everything in this function is 3d

    schmidTensor = getSchmidTensor(index)
    ### Need to verify what equation is correct: report or paper ###

    #### ANDREW's DEBUGGING #2: this is the implementation for calculating tau_s that i found in moose."CrystalSlipRatePlasticityGSS.C"
    #### This seems to agree with Nicolo's paper. 
    tau_s = np.tensordot(S,schmidTensor,axes=2)
    # tau_s = np.tensordot(np.dot(F.transpose(), np.dot(F,S)), schmidTensor,axes=2) 




    #print("F",F)  # close to identity matrix
    #print("S",S)  # composed of e7 values, some times all values close to 0
    # print("schmidTensor",schmidTensor) # composed of 0,1,-1
    # print("g_prev",g_si)                 # 1   
    # print("gamma_dot_ref",gamma_dot_ref) # 1.0*e7
    # print("tau_s",tau_s) # +-e8 or 0 some cases

    tau_th_s = getStrengthRatio(index) * g_si
    # print("tau_th_s",tau_th_s)
    # print("g_si",g_si)
    slipRate = gamma_dot_ref * np.sign(tau_s) * np.abs(tau_s/tau_th_s)**(1/m)
    if np.isnan(slipRate):
        time.sleep(100)
    
    #print("slipRate",slipRate) # sliprate is huge e80 for now
    # input("press enter")

    # print('S',S)
    # print('tau_s', tau_s)
    # print('tau_th_s',tau_th_s)

    return slipRate, schmidTensor,tau_th_s,tau_s

def getNextResistance(g_sat, g_prev, a, h, slipRates, dt):

    g_dot = np.zeros([10,1])


    for i in range(10):
        # Andrew's debugging: the next resistance must take into account the slip from all slip planes.
        for j in range(10):

            g_dot[i] += h * (1-g_prev[j]/g_sat)**a * slipRates[j]

    # Time discretization of ODE
    g_current = g_prev + dt*g_dot

    
    return g_current




def getSchmidTensor(index):
    
    slipDirection, slipPlane = getSlipSystems(index)

    schmidTensor = np.tensordot(slipDirection, slipPlane, axes = 0) # axes = 0 means tensor product
    # print("schmidTensor, ",schmidTensor) # size of 3*3
    return schmidTensor

def getStrengthRatio(index):

    strengthRatios = np.array([1,\
    0.963,\
    0.963,\
    0.933,\
    1.681,\
    0.376,\
    0.931,\
    0.931,\
    0.701,\
    0.701])

    strengthRatio = strengthRatios[index]

    return strengthRatio


def getSlipSystems(index):

    # 10*3 dimension
    slipDirections = \
    np.array([ \
    [1.,0.,0.], [1.,0.,0.], \
    [-1.,0.,0.], [0.,1.,0.],\
    [1.,0.,0.], [2.,0.,1.], \
    [0.,-1.,1.], [0.,-1.,-1.],\
    [-1.,0.,-1.], [1.,0.,1.] ])

    # 10*3 dimension
    slipPlanes = \
    np.array([ \
    [0.,1.,0.], [0.,1.,1.],\
    [0.,1.,-1.], [-1.,0.,2.], \
    [0.,0.,1.], [-1.,0.,2.], \
    [0.,1.,1.], [0.,-1.,1.], \
    [-1.,-1., 1.], [1.,-1.,-1.] ])

    slipPlane = slipPlanes[index]
    slipDirection = slipDirections[index]

    return slipDirection, slipPlane

def calcDFpDSe(dt,  F_p_current, gamma_dot_ref, g_current,m, S_elastic):
    Fp = np.zeros([3,3])
    Fp[0:2,0:2] = F_p_current
    Fp[2,2] = 1

    S_e = np.zeros([3,3])
    S_e[0:2,0:2] = S_elastic
    S_e[2,2] = 1

    dFpdSe = np.zeros([2,2,2,2])
    for alpha_i in range(10):
        schmid = getSchmidTensor(alpha_i)
        dFpdgamma = dt* np.dot(schmid,Fp)

        tau_alpha = np.tensordot(S_e,schmid,axes= 2)
        tau_th_alpha = getStrengthRatio(alpha_i) * g_current[alpha_i]
        dgammadtau = gamma_dot_ref/m/tau_th_alpha * np.abs(tau_alpha/tau_th_alpha)**(1/m-1)

        dtaudSe = schmid

        placeholder = np.tensordot(dFpdgamma, dgammadtau * dtaudSe, axes = 0)

        dFpdSe += placeholder[0:2,0:2, 0:2, 0:2] # turn into a 4th order 2d

    return dFpdSe





# ANDREW: Created this function to check for convergence of slip resistance parameters for each slip plane.
def notConvergedYet(g_prev,g_current, g_tol):
    
    numSlipPlanes = np.shape(g_prev)[0]
    
    for i in range(numSlipPlanes):
        diff = np.abs(g_prev[i] - g_current[i])

        if diff > (np.abs(g_prev[i])*g_tol):
            print('Comparison',g_prev[i],g_current[i])
            return True


    return False