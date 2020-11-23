import numpy as np 
def computeSecondPiola(F_e,const_dictionary):
    # Parameters we use in function
    Gamma = const_dictionary["Gamma"] # Mie Gruneisen Parameter []
    T = const_dictionary["T"] # Ambient temperature of material [K]
    T_0 = const_dictionary["T_0"] # Reference temperature of material [K]
    v = const_dictionary["v"] # (ARBITRARY, something less that v_0) Specific volume of material at temp T [m^3/Kg]
    rho_0 = const_dictionary["rho_0"] # Initial density [Kg/m^3] 
    v_0 = const_dictionary["v_0"] # specific volume of material at reference temp [m^3/Kg]
    K_0 = const_dictionary["K_0"] # Reference bulk modulus [Pa]
    C_v = const_dictionary["C_v"] # Specific heat capacity [J/(Kg*K)]
    s = const_dictionary["s"] # Slope Hugoniot []
    alpha = const_dictionary["alpha"] # Thermal expansion tensor []
    gamma_dot_ref = const_dictionary["gamma_dot_ref"] # Reference slip rate [s]
    m = const_dictionary["m"] # Slip rate exponent []
    g = const_dictionary["g"]
    g_sat = const_dictionary["g_sat"]# Saturation slip resistance [Pa]
    a = const_dictionary["a"] # Hardening exponent []
    h = const_dictionary["h"]# Hardening matrix [Pa]
    C_elastic = const_dictionary["C_ela"]


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

    S=S_el+S_eos  # vis dropped for now
    return S