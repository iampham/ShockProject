def calculateNextPlastic(F_p,gamma_dot_ref, m, g, g_sat, g_prev, a, h, dt, F_e):
    
    
    result_inc, g_next=calculateResultantIncrement(gamma_dot_ref, m, g, g_sat, g_prev, a, h, dt, F_e)

    F_p_next=np.dot(result_inc,F_p)

    return F_p_next,g_next

def calculateResultantIncrement(gamma_dot_ref, m, g, g_sat, g_prev, a, h, dt, F_e):
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

    F = np.zeros([3,3])
    # 4 - Fixed this should 0:2, not 0:1
    F[0:2,0:2] = F_e
    F[2,2] = 1

    strainIncrement = np.zeros([3,3])

    slipRates = np.zeros([10,1])

    for index in range(10):

        slipRates[index] = calculateSlipRate(gamma_dot_ref, m, g[index], index)
        strainIncrement += slipRates[index] * schmidTensor
    
    resultant_increment=np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i==j:
                resultant_increment[i,j] += 1 - strainIncrement[i,j]
            else:
                resultant_increment[i,j] += 0 - strainIncrement[i,j]


    g_next = getNextResistance(g_sat, g, a, h, slipRates, dt)

    return resultant_increment[0:2, 0:2], g_next

def calculateSlipRate(gamma_dot_ref, m, g_si, index):

    schmidTensor = getSchmidTensor(index)
    tau_s = np.dot(np.matmul(F.transpose, np.matmul(F,S)), schmidTensor) 

    tau_th_s = getStrengthRatio(index) * g_si
    slipRate = gamma_dot_ref * sign(tau_s) * np.abs(tau_s/tau_th_s)**(1/m)

    return slipRate

def getNextResistance(g_sat, g_prev, a, h, slipRates, dt):

    g_dot = np.zeros([10,1])


    for i in range(10):

        g_dot[i] = h * (1-g_prev[i]/g_sat)**a * slipRates[i]


    g_current = g_prev + dt*g_dot
    
    return g_current




def getSchmidTensor(index):
    
    slipDirection, slipPlane = getSlipSystems(index)

    schmidTensor = np.tensordot(slipDirection, slipPlane, axes = 0) # axes = 0 means tensor product

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

    strengthRatio = strengthRatios[i]

    return strengthRatio


def getSlipSystems(index):

    slipDirections = \
    np.array([ \
    [1.,0.,0.], [1.,0.,0.], \
    [-1.,0.,0.], [0.,1.,0.],\
    [1.,0.,0.], [2.,0.,1.], \
    [0.,-1.,1.], [0.,-1.,-1.],\
    [-1.,0.,-1.], [1.,0.,1.] ])

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