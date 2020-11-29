import numpy as np 
def computeSecondPiola(F_e,const_dictionary,v):
    # Parameters we use in function
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
    C_elastic = const_dictionary["C_ela"]

    # # TODO: debugging, force F to I
    # F_e=np.eye(2)*0.99

    J = np.linalg.det(F_e)

    C_e = np.dot(F_e.transpose(), F_e)



    I = np.eye(2)
    # print(np.dot(F_e.transpose(), F_e))
    E_e = 1/2 * (np.dot(F_e.transpose(), F_e) - J**(2/3) * I )

    
    strainTerm=  E_e - alpha*(T-T_0)
    strainTerm_voigt = np.array([[strainTerm[0,0], strainTerm[1,1], 2*strainTerm[0,1]]])
    strainTerm_voigt = strainTerm_voigt.transpose()
    
    S_el_voigt = np.dot(C_elastic, strainTerm_voigt)

    S_el = np.zeros([2,2])
    S_el[0,0] = S_el_voigt[0]
    S_el[1,1] = S_el_voigt[1]
    S_el[0,1] = S_el_voigt[2] 
    S_el[1,0] = S_el_voigt[2] 
    

    chi = 1 - v/v_0
    p_eos = Gamma* rho_0 * C_v * (T-T_0)* (v_0/v) + K_0*chi/(1-s*chi)**2 * (Gamma/2 * (v_0/v - 1) - 1)
    S_eos = -J * p_eos * np.linalg.inv(C_e)

    # TODO : debugging, make eos term 0
    S_eos=np.zeros([2,2])

    S=S_el+S_eos  # vis dropped for now
    
    return S,S_eos,S_el_voigt,p_eos

def computeSecondPiolaJacobian(S_prev,F_p_prev,F,C_elastic):


    # Plastic deformation inverse
    F_p_inv_prev = np.linalg.inv(F_p_prev)
    # Elastic deformation gradient
    F_e_prev = np.dot(F,F_p_inv)

    def KronDel(m,n):
        if m==n:
            delta = 1
        else: 
            delta = 0
        return delta

    # 4th order identity
    II = np.zeros((2,2,2,2))
    # Other components of tangent
    dFedFpinv = np.zeros([2,2,2,2])
    dEedFe = np.zeros([2,2,2,2])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    II[i,j,k,l] = KronDel(i,k)*KronDel(j,l)
                    dFedFpinv[i,j,k,l] = F[i,k]*KronDel(l,j)
                    dEedFe[i,j,k,l] = (0.5*KronDel(i,l)*F_e_prev[k,j]+0.5*KronDel(j,l)*F_e_prev[k,i])


    dFpinvdDResultantInc = F_p_inv_prev
    dDResultantIncdSlipRate = np.zeros([3,3])
    # TODO need to make dimensions agree (dDResultantIncdSlipRate should be a 2x2)
    for s_i in range(10):
        dDResultantIncdSlipRate -= getSchmidTensor(s_i)

    # TODO need to compute the last 2 terms of the tangent
    dSlipRatedtao = 
    dtaodSprev = 

    # Compute Second Piola Kirchoff tangent
    # TODO need right operations in between variables
    S_tangent (2,2,2,2)= dEedFe(2,2,2,2) : dFedFpinv(2,2,2,2) : dFpinvdDResultantInc(2,2) : dDResultantIncdSlipRate(3,3) : dSlipRatedtao : dtaodSprev

    # Jacobian of Second Piola Kirchoff
    J_S = II - np.tensordot(C_elastic,S_tangent,axes=2)

    return J_S