def assembleRRKK(Gamma, T, T_0, v, v_0, K_0, rho_0, C_v, s, alpha):
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
            F_p = calculateNextPlastic(F_p,gamma_dot_ref, m, g, g_sat, g_prev, a, h, dt, F_e)# TODO: Calculate the plastic part of the deformation tensor
            F_e = F * np.linalg.inv(F_p)


            detF_e = np.linalg.det(F_e)
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
            E_e = 1/2 * (np.dot(F_e.transpose(), F_e) - detF_e ** (2/3) * I )

            S_el = np.tensordot(C_elastic, (E_e - alpha * (T-T_ref)))

            chi = 1 - v/v0
            p_eos = Gamma* rho_0 * C_v * (T-T_0)* (v_0/v) + K_0*chi/(1-s*chi)**2 * (Gamma/2 * (v_0/v - 1) - 1)
            S_eos = -detF_e * p_eos * np.linalg.inv(C_e)

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
                    Re[ni*2+ci] += wi*np.linalg.det(dxdxi)*np.tensordot(S,deltaE)

                    # ASSEMBLE INTO GLOBAL RESIDUAL (I didn't ask for this)
                    RR[node_ei[ni]*2+ci] += wi*np.linalg.det(dxdxi)*np.tensordot(S,deltaE)
                    
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
                            Kmat_el = np.dot(C_voigt, Deltaeps_voigt)
                            F_e_inv=np.linalg.inv(F_e)
                            Kmat_eos= -p_eos *  (np.dot(J*F_e_inv.transpose(),gradX_Du)) + 
                            Kmat = np.dot(Deltaeps_voigt,np.dot(D,deltad_voigt))
                            # add to the corresponding entry in Ke and dont forget other parts of integral
                            Ke[ni*2+ci,nj*2+cj] += wi*np.linalg.det(dxdxi)*(Kgeom+Kmat)
                            # assemble into global 
                            KK[node_ei[ni]*2+ci,node_ei[nj]*2+cj] += wi*np.linalg.det(dxdxi)*(Kgeom+Kmat)
                            
    return RR,KK
                            
        