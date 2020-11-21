def calculatePlastic(gamma_dot_ref, slip_exp, q_si, q_sat, h_0, q_sisj,  F_p_prev):
    """ 
    Calculates the plastic part of the deformation tensor. Uses the following equation:
    F_p = deltaGamma dot F_p_prev
    deltaGamma = 
    

    INPUTS: 
    gamma_dot: reference slip rate
    slip_exp: slip rate exponent

    g_si 
    g_sat
    h_0
    q_sisj


    """ 


    
    return F_p


def getSchmidTensor(index):
    
    slipDirection, slipPlane = getSlipSystems(index)

    schmidTensor = np.tensordot(slipDirection, slipPlane, axes = 0) # axes = 0 means tensor product

    return schmidTensor

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