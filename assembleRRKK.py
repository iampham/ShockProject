import numpy as np 
from calculatePlastic import * 

def assembleRRKK(const_dictionary,Nvec, dNvecdxi, n_node, n_elem, elements, node_X, node_x):
    """
    INPUTS: State & Material parameters to the equation:
        - Gamma: Mie Gruneisen Parameter
        - T: Ambient temperature of material
        - T_0: Reference temperature of material
        - v: specific volume of material at temp T
        - v_0: specific volume of material at reference temp
        - K_0: Reference bulk modulus
        - rho_0: Initial density
        - C_v: Specific heat capacity
        - s: Slope Hugoniot
        - alpha: Thermal expansion tensor


    """
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
    g_prev = const_dictionary["g_prev"] # NEED CORRECT VALUE [Pa]
    a = const_dictionary["a"] # Hardening exponent []
    h = const_dictionary["h"]# Hardening matrix [Pa]
    C_elastic = const_dictionary["C_ela"]

    # assemble total residual 
    RR = np.zeros((n_node*2))
    # assemble the total tangent 
    KK = np.zeros((n_node*2,n_node*2))
    # loop over elements
    for ei in range(n_elem): 
        # initialize the residual for this element
        Re = np.zeros((8))
        # initialize the tangent for this element
        Ke = np.zeros((8,8))

        # nodes that make up this element 
        node_ei = elements[ei]
        # reference coordinates of the nodes making up this element (init to zero, fill in a loop)
        node_X_ei = np.zeros((4,2))
        # deformed coordinates of the nodes making up this element (init to zero, fill in a loop)
        node_x_ei = np.zeros((4,2))
        for ni in range(4):
            node_X_ei[ni] = node_X[node_ei[ni]]
            node_x_ei[ni] = node_x[node_ei[ni]]

        # also, do a proper integration with four integration points 
        # Loop over integration points
        # location and weight of integration points 
        IP_xi = np.array([[-1./np.sqrt(3),-1./np.sqrt(3)],[+1./np.sqrt(3),-1./np.sqrt(3)],\
                          [+1./np.sqrt(3),+1./np.sqrt(3)],[-1./np.sqrt(3),+1./np.sqrt(3)]])
        IP_wi = np.array([1.,1.,1.,1.])
        for ip in range(4):
            xi  = IP_xi[ip,0]
            eta = IP_xi[ip,1]
            wi = IP_wi[ip]
            # eval shape functions 
            Ns = Nvec(xi,eta)
            # eval the isoparametric map for the reference and deformed points corresponding to xi,eta = 0
            X = np.zeros((2))
            x = np.zeros((2))
            for ni in range(4):
                X += Ns[ni]*node_X_ei[ni]
                x += Ns[ni]*node_x_ei[ni]

            # evaluate the Jacobians, first derivative of shape functions with respect to xi space then Jacobians 
            dNsdxi = dNvecdxi(xi,eta)
            dXdxi = np.zeros((2,2))
            dxdxi = np.zeros((2,2))
            for ni in range(4):
                dXdxi += np.outer(node_X_ei[ni],dNsdxi[:,ni])
                dxdxi += np.outer(node_x_ei[ni],dNsdxi[:,ni])

            # get gradient of basis function with respect to X using inverse jacobian 
            JinvT = np.linalg.inv(dXdxi).transpose()
            dNsdX = np.dot(JinvT,dNsdxi)

            # get gradient of basis function with respect to x using inverse jacobian, the other one 
            jinvT = np.linalg.inv(dxdxi).transpose()
            dNsdx = np.dot(jinvT,dNsdxi)

            # get the deformation gradient 
            F = np.zeros((2,2))
            for ni in range(4):
                F += np.outer(node_x_ei[ni],dNsdX[:,ni])
            # compute the stress, some parameters not defined
            
            # 1 - Need initial value of F_p and F_e
            F_e_0 = F
            F_p_0 = np.eye(2)
            # 2 - Need values for internal functions




            dt = 1
            
            # Iteration to calculate actual split of F=F_p*F_e
            S_0  = np.eye(2)
            F_p, g_next = calculateNextPlastic(F_p_0,gamma_dot_ref, m, g_prev, g_sat, g_prev, a, h, dt, F_e_0, S_0)# TODO: Calculate the plastic part of the deformation tensor
            F_e = F * np.linalg.inv(F_p)


            J = np.linalg.det(F_e)
            C_e = np.dot(F_e.transpose(), F_e)
            # special dyadic product of C_e
            C_e_inv = np.linalg.inv(C_e)
            # C_inv Special Dyad C_inv
            dyad_bar = np.zeros([2,2,2,2])
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        for l in range(2):
                            dyad_bar[i,j,k,l] = -C_e_inv[i,k]*C_e_inv[j,l] 

            I = np.eye(2)
            E_e = 1/2 * (np.dot(F_e.transpose(), F_e) - J ** (2/3) * I )

            
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


            

            # compute the variation of the symmetric velocity gradient by moving one node and one component
            # of that node at a time, except if the node is on the boundary in which case no variation is allowed
            for ni in range(4): # deltav and deltau corresponds to a,b in lecture, ni is # nodes in elem
                deltav = np.zeros((2))
                for ci in range(2): # coord in x and y
                    # note, no worries about the boundary because we will get rid of the corresponding rows
                    # of the residual because they wont be zero 
                    deltav[ci] = 1
                    gradX_v = np.outer(deltav,dNsdX[:,ni])
                    deltaE= 0.5*(np.dot(F.transpose(),gradX_v) + np.dot(gradX_v.transpose(),F)  )
                    Re[ni*2+ci] += wi*np.linalg.det(dXdxi)*np.tensordot(S,deltaE)

                    # ASSEMBLE INTO GLOBAL RESIDUAL (I didn't ask for this)
                    RR[node_ei[ni]*2+ci] += wi*np.linalg.det(dXdxi)*np.tensordot(S,deltaE)
                    
                    ## 2 more for loops for the increment Delta u
                    for nj in range(4):
                        Deltau = np.zeros((2))
                        for cj in range(2):
                            Deltau[cj]=1
                            gradX_Du = np.outer(Deltau,dNsdX[:,nj])
                            # 
                            Deltaeps = 0.5*(np.dot(gradX_Du.transpose(),F_e) + np.dot(F_e.transpose(),gradX_Du))
                            
                            ## ELEMENT TANGENT
                            # Initial stress component (also called geometric component) is 
                            # refer to Linearization.pdf
                            Kgeom = np.tensordot(S,np.dot(gradX_v.transpose(),gradX_Du) + np.dot(gradX_Du.transpose(),gradX_v))
                            # Material component, need to put things in voigt notation for easy computation
                            # deltad_voigt = np.array([deltad[0,0],deltad[1,1],2*deltad[0,1]])
                            
                            # D = np.array([[4*p,2*p,0],[2*p,4*p,0],[0,0,2*p]])
                            Deltaeps_voigt = np.array([Deltaeps[0,0],Deltaeps[1,1],2*Deltaeps[0,1]])
                            Kmat_el_voigt = np.dot(C_elastic, Deltaeps_voigt)

                            print(np.shape(Kmat_el_voigt))

                            Kmat_el = np.zeros([2,2])
                            Kmat_el[0,0] = Kmat_el_voigt[0]
                            Kmat_el[1,1] = Kmat_el_voigt[1]
                            Kmat_el[0,1] = Kmat_el_voigt[2]
                            Kmat_el[1,0] = Kmat_el_voigt[2]

                            F_e_inv=np.linalg.inv(F_e)
                            DC_e=np.dot(gradX_Du.transpose(),F) + np.dot(F.transpose(),gradX_Du)
                            Kmat_eos= -p_eos *  (np.tensordot(J*F_e_inv.transpose(),gradX_Du,axes=2)) + np.tensordot(dyad_bar,DC_e)

                            print('Kmat_eos',np.shape(Kmat_eos))
                            Kmat = Kmat_el + Kmat_eos
                            # add to the corresponding entry in Ke and dont forget other parts of integral

                            print(np.shape(wi*np.linalg.det(dXdxi)*(Kgeom+Kmat)))
                            print(np.shape(Kgeom))
                            print(np.shape(Kmat))
                            Ke[ni*2+ci,nj*2+cj] += wi*np.linalg.det(dXdxi)*(Kgeom+Kmat)
                            # assemble into global 
                            KK[node_ei[ni]*2+ci,node_ei[nj]*2+cj] += wi*np.linalg.det(dXdxi)*(Kgeom+Kmat)
                            
    return RR,KK