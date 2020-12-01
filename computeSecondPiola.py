import numpy as np
from calculatePlastic import * 
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
    C_ela_2d_voigt = const_dictionary["C_ela_2d_voigt"]
    C_ela_3d=const_dictionary["C_ela_3d"]

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
    
    S_el_voigt = np.dot(C_ela_2d_voigt, strainTerm_voigt)

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

def computeSecondPiolaResidualJacobian(S_prev,F_p_prev,F,g,dt,const_dictionary):
    # Assume S,FP,F are 3*3 matrices
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
    C_ela_2d_voigt = const_dictionary["C_ela_2d_voigt"]
    C_ela_3d=const_dictionary["C_ela_3d"]

    # Go to 3D
    S_prev_3D = np.zeros([3,3])
    S_prev_3D[0:2,0:2] = S_prev
    F_p_prev_3D = np.eye(3)
    F_p_prev_3D[0:2,0:2] = F_p_prev
    F_3D = np.eye(3)
    F_3D[0:2,0:2] = F

    # Plastic deformation inverse
    F_p_inv_prev = np.linalg.inv(F_p_prev_3D)
    # Elastic deformation gradient
    F_e_prev = np.dot(F_3D,F_p_inv_prev)

    # Compute Residual
    S_prime = np.tensordot(C_ela_3d,0.5*(np.dot(F_e_prev.transpose(),F_e_prev)-np.eye(3)),axes=2)
    res_S = S_prev_3D - S_prime

    def KronDel(m,n):
        if m==n:
            delta = 1
        else: 
            delta = 0
        return delta

    # 4th order identity
    II = np.zeros([3,3,3,3])
    # Other components of tangent
    dFedFpinv = np.zeros([3,3,3,3])
    dEedFe = np.zeros([3,3,3,3])
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    II[i,j,k,l] = KronDel(i,k)*KronDel(j,l)
                    dFedFpinv[i,j,k,l] = F_3D[i,k]*KronDel(l,j)
                    dEedFe[i,j,k,l] = (0.5*KronDel(i,l)*F_e_prev[k,j]+0.5*KronDel(j,l)*F_e_prev[k,i])


    #  F_p_inv_prev to be (3x3)
    dFpinvdDResultantInc = np.zeros([3,3])
    dFpinvdDResultantInc = F_p_inv_prev

    dDResultantIncdSlipRate = np.zeros([3,3])
    dFpinvdDslip=np.zeros([3,3])
    dtaudSprev = np.zeros([3,3])
    dSlipRatedtau = np.zeros([10,1])
    
    dFpinvdSprev=np.zeros([3,3,3,3])
    # Iterating for all slip systems alpha  
    for alpha_i in range(10):
        dDResultantIncdSlipRate = - getSchmidTensor(alpha_i)
        dtaudSprev = getSchmidTensor(alpha_i)
        dFpinvdDslip = np.dot(dFpinvdDResultantInc,dDResultantIncdSlipRate)

        # get g_si
        g_si=g[alpha_i]
        tau,schmid,tau_th=calculateSlipRate(F_3D,S_prev_3D,gamma_dot_ref, m, g_si, alpha_i)
        dSlipRatedtau[alpha_i]=gamma_dot_ref / m * (np.abs(tau/tau_th)**(1/m-1.))/tau_th # scalar

        # calculate dFpinvdSprev by combining terms above
        dFpinvdSprev += np.tensordot(dFpinvdDslip * dSlipRatedtau[alpha_i] * dt, dtaudSprev,axes=0)

    
    # Compute Second Piola Kirchoff tangent
    S_tangent = np.tensordot(dEedFe,np.tensordot(dFedFpinv,dFpinvdSprev,axes=2),axes=2)

    # Jacobian of Second Piola Kirchoff
    J_S = II- np.tensordot(C_ela_3d,S_tangent,axes=2)

    return res_S, J_S