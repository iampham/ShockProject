import numpy as np 
from calculatePlastic import * 
from computeSecondPiola import *

def assembleRRKK(const_dictionary,Nvec, dNvecdxi, n_nodes, n_elem, elements, node_X, node_x,F_p_prev,g_prev,dt):
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
    g_sat = const_dictionary["g_sat"]# Saturation slip resistance [Pa]
    a = const_dictionary["a"] # Hardening exponent []
    h = const_dictionary["h"]# Hardening matrix [Pa]
    C_ela_2d_voigt = const_dictionary["C_ela_2d_voigt"]
    C_ela = const_dictionary["C_ela_2d"]
    n_IP=const_dictionary["n_IP"]

    # assemble total residual 
    RR = np.zeros((n_nodes*2))
    # assemble the total tangent 
    KK = np.zeros((n_nodes*2,n_nodes*2))
    # F_p global for all elements in the mesh
    F_p_next=np.zeros([2,2,n_elem,n_IP])
    # g_next global for all elements in mesh
    g_next=np.zeros([10,1,n_elem,n_IP])
    sigma_next=np.zeros([2,2,n_elem,n_IP])
    S_next_all=np.zeros([2,2,n_elem,n_IP])
    F_next=np.zeros([2,2,n_elem,n_IP])
    F_e_next=np.zeros([2,2,n_elem,n_IP])
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
        for ip in range(n_IP):
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
            F_p_prev_loc = F_p_prev[:,:,ei,ip]
            # calculate F_e from F and F_p
            F_e_prev_loc = np.dot(F,np.linalg.inv(F_p_prev_loc)) 

            # compute strain variables before interations
            C_e=np.dot(F_e_prev_loc.transpose(),F_e_prev_loc)
            C_e_inv=np.linalg.inv(C_e)
            I = np.eye(2)
            E_e = 0.5*(C_e-I)


            # Slip resistance stop parameter We don't actually use this
            g_tol = 1000

            g_itermax = 1
            g_iter = 0
            g_not_converged = True
            g_prev_loc=g_prev[:,:,ei,ip]


            while (g_not_converged and g_iter<g_itermax):
                # according to the current shockwave boundary and coordinates of ip, update v
                S_prev,S_eos,S_el_voigt=computeSecondPiola(F_e_prev_loc,const_dictionary,v,C_e,C_e_inv,E_e)

                # Initialize stress residual
                res_S = 1.
                norm_res_S = 1
                S_iter = 0
                S_tol = 1e-5
                S_itermax = 1000

                # Solve Stress newton raphson
                while (norm_res_S>S_tol and S_iter<S_itermax):

                
                    # according to the current shockwave boundary and coordinates of ip, update v
                    
                    # Residual and Jacobian to compute Next Stress                    
                    res_S, J_S, F_e_current = computeSecondPiolaResidualJacobian(S_prev,F_p_prev_loc,F,g_prev_loc,dt,const_dictionary) 
                    F_e_current=F_e_current[0:2,0:2]
                    # compute delta_S and add it to S
                    # slice 3d to 2d before the tensordot and increment 
                    J_S=J_S[0:2,0:2,0:2,0:2]
                    # print("J_S",J_S)
                    res_S=res_S[0:2,0:2]
                    delta_S = -np.tensordot(np.linalg.tensorinv(J_S),res_S,axes=2)
                    
                    S_current = S_prev + delta_S[0:2,0:2]
                    S_current = S_current[0:2,0:2] # Go back to 2D
                    # Stop criteria
                    norm_res_S = np.linalg.norm(res_S)


                    # # TODO Need to add shock condition to next guess of S_prev
                    # if x[0]<shock_bound:
                    #     S_prev,S_eos,S_el_voigt,p_eos=computeSecondPiola(F_e_prev_loc,const_dictionary,v)
                    # else:
                    #     S_prev,S_eos,S_el_voigt,p_eos=computeSecondPiola(F_e_prev_loc,const_dictionary,v_0)
                    # # Iteration to calculate actual split of F=F_p*F_e

                    S_prev = S_current
                    S_iter += 1
                    # print("S_iter ",S_iter)
                    # print("res_S",norm_res_S)
                    # print(norm_res_S>S_tol)
                    # print("S:" , S_current)
                    # end of S loop

                    
                F_p_current, g_current = calculateNextPlastic(F_p_prev_loc,gamma_dot_ref, m, g_sat, g_prev_loc, a, h, dt, F_e_current, S_current)
                # print("F_p_current",F_p_current)
                g_diff = np.abs(g_prev_loc-g_current)
                # if g_diff>g_max:
                #     g_max = g_diff
                


                g_not_converged = notConvergedYet(g_prev_loc,g_current,g_tol)
                g_iter += 1
                # print("g_iter",g_iter)
                # print("g_condition",g_iter<g_itermax)

                # TODO we are getting a huge F_p (1e80+) and that makes F_e really small
                # print("F_p_prev_loc",F_p_prev_loc,"gamma_dot_ref",gamma_dot_ref,"m",m, "g_sat",g_sat, "g_prev_loc",g_prev_loc,\
                #       "a",a, "h",h, "dt",dt, "F_e_prev_loc",F_e_prev_loc, "S_prev",S_prev)

                # # F_p = np.eye(2) # TODO 27 nov 2020 debugging proved that elastic part is working properly, still need to figure out how to make the plastic part work
                # F_e = np.dot(F,  np.linalg.inv(F_p))            
                # # print("F",F)
                # # print("F_p",F_p)
                # # print("F_e",F_e)
                # end of g loop

            S_elastic = S_current

            # print("outside")
            F_p_inv = np.linalg.inv(F_p_current)
            F_p_invT = F_p_inv.transpose()
            S_all = np.dot(F_p_inv,np.dot(S_elastic, F_p_invT))
            print("F_p_inv",F_p_inv)
            print("S_elastic",S_elastic)
            print("S_all",S_all)

            S_all_voigt = np.zeros([1,3])
            S_all_voigt[0,0] = S_all[0,0]
            S_all_voigt[0,1] = S_all[1,1]
            S_all_voigt[0,2] = 2 * S_all[0,1]


            # update the variables with new F_e
            J_e = np.linalg.det(F_e_current)
            C_e = np.dot(F_e_current.transpose(), F_e_current)
            # special dyadic product of C_e
            C_e_inv = np.linalg.inv(C_e)
            C_e_inv=np.linalg.inv(C_e)
            I = np.eye(2)
            E_e = 0.5*(C_e-I)

            J = np.linalg.det(F)
            C = np.dot(F.transpose(), F)
            Cinv = np.linalg.inv(C)
            E = 0.5*(C-I)



            # # C_inv Special Dyad C_inv
            # dyad_bar = np.zeros([2,2,2,2])
            # for i in range(2):
            #     for j in range(2):
            #         for k in range(2):
            #             for l in range(2):
            #                 dyad_bar[i,j,k,l] = C_e_inv[i,k]* F_p_inv[j,l] 

            
            

            # if x[0]<shock_bound:
            #     S,S_eos,S_el_voigt,p_eos=computeSecondPiola(F_e,const_dictionary,v)
            # else:
            #     S,S_eos,S_el_voigt,p_eos=computeSecondPiola(F_e,const_dictionary,v_0)       

    
            # F_p_current = np.dot(F,np.linalg.inv(F_p))            
            # F_p_prev_loc = F_p_prev[:,:,ei,ip]
            # calculate F_e from F and F_p
            # F_e_prev_loc = np.dot(F,np.linalg.inv(F_p_prev_loc)) 

            # Post Solve Stress

            # Compute Cauchy with actual value of elastic deformation
            # J = np.linalg.det(F)
            # sigma = (1/J)*np.dot(F_e_current,np.dot(S_current,F_e_current.transpose()))
            
            # store results in global variables
            g_next[:,:,ei,ip] = g_current
            # sigma_next[:,:,ei,ip] = sigma
            S_next_all[:,:,ei,ip] = S_current
            F_e_next[:,:,ei,ip] = F_e_current
            F_p_next[:,:,ei,ip] = F_p_current
            F_next[:,:,ei,ip] = F

            # compute the variation of the symmetric velocity gradient by moving one node and one component
            # of that node at a time, except if the node is on the boundary in which case no variation is allowed
            for ni in range(4): # deltav and deltau corresponds to a,b in lecture, ni is # nodes in elem
                for ci in range(2): # coord in x and y
                    # note, no worries about the boundary because we will get rid of the corresponding rows
                    # of the residual because they wont be zero 
                    deltav = np.zeros((2))
                    deltav[ci] = 1
                    gradX_v = np.outer(deltav,dNsdX[:,ni])
                    deltaE= 0.5*(np.dot(F.transpose(),gradX_v) + np.dot(gradX_v.transpose(),F))
                    deltaE_voigt = np.array([deltaE[0,0],deltaE[1,1],2*deltaE[0,1]])
                    # Re[ni*2+ci] += wi*np.linalg.det(dXdxi)*np.tensordot(S_current,deltaE)

                    # ASSEMBLE INTO GLOBAL RESIDUAL (I didn't ask for this)
                    RR[node_ei[ni]*2+ci] += wi*np.linalg.det(dXdxi)*np.tensordot(S_all,deltaE)

                    
                    ## 2 more for loops for the increment Delta u
                    for nj in range(4):
                        for cj in range(2):
                            Deltau = np.zeros((2))
                            Deltau[cj]=1
                            gradX_Du = np.outer(Deltau,dNsdX[:,nj])
                            Delta_delta_E = 0.5*np.dot(gradX_v.transpose(),gradX_Du)+\
                                np.dot(gradX_Du.transpose(),gradX_v)

                            Delta_delta_E_voigt = np.array([[Delta_delta_E[0,0],Delta_delta_E[1,1],2*Delta_delta_E[0,1]]])
                            
                            Deltaeps = 0.5*(np.dot(gradX_Du.transpose(),F) + np.dot(F.transpose(),gradX_Du)) # ANDREW: Changed from F_e_current to F
                            
                            ## ELEMENT TANGENT
                            # Initial stress component (also called geometric component) is 
                            # refer to Linearization.pdf
                            Kgeom = np.tensordot(S_all, Delta_delta_E,axes=2)
                            # Material component, need to put things in voigt notation for easy computation
                            # deltad_voigt = np.array([deltad[0,0],deltad[1,1],2*deltad[0,1]])
                            
                            # D = np.array([[4*p,2*p,0],[2*p,4*p,0],[0,0,2*p]])
                            Deltaeps_voigt = np.array([Deltaeps[0,0],Deltaeps[1,1],2*Deltaeps[0,1]])
                            # First terms of elastic material part

                            F_p_inv_up = specialDyadUp(F_p_inv, F_p_inv)
                            F_p_invT_up = specialDyadUp(F_p_invT, F_p_invT)

                            F_p_inv_up_S = specialDyadUp(F_p_inv, S_all)
                            S_all_down_F_p_inv = specialDyadDown(S_all, F_p_inv)


                            F_p_invT_down_C_e = specialDyadDown(F_p_invT, C_e)
                            C_e_up_F_p_invT = specialDyadUp(C_e, F_p_invT)

                            # F_p_dyad_down = specialDyadDown(F_p_current, F_p_current)

                            #### Part 1 of material stiffness
                            dSdC = np.tensordot(F_p_inv_up, np.tensordot(C_ela,F_p_invT_up, axes = 2), axes = 2)
                            # print("dSdC_norm",np.linalg.norm(dSdC))
                            # print("Fpinvt", np.linalg.norm(F_p_inv_up))
                            

                            #### Part 2 of material stiffness 
                            dSdFp = -(F_p_inv_up_S + S_all_down_F_p_inv) - np.tensordot((F_p_inv_up), \
                                            np.tensordot(0.5*C_ela, F_p_invT_down_C_e + C_e_up_F_p_invT, axes =2),  axes = 2) # large term 
                            # print("dSdFp_norm",np.linalg.norm(dSdFp))

                            dFpdSe = calcDFpDSe(dt,  F_p_prev_loc, gamma_dot_ref, g_current,m, S_elastic)
                            # print("dFpdSe_norm",np.linalg.norm(dFpdSe))
                            dSedCe = 0.5 *C_ela # large term
                            # print("dSedCe_norm",np.linalg.norm(dSedCe))
                            dCedC = np.tensordot(F_p_invT, F_p_inv, axes = 0) 
                            # print("dCedC_norm",np.linalg.norm(dCedC))
                            

                            bigdaddy = np.tensordot(dSdFp, \
                                            np.tensordot(dFpdSe,\
                                                np.tensordot(dSedCe, dCedC, axes = 2), axes = 2), axes =2)

                            
                            # print( "dCedC",  np.linalg.norm(dCedC))

                            bigC = 2 * dSdC + 2 * bigdaddy 
                            # print("bigdaddy_norm",np.linalg.norm(bigdaddy))

                            Kmat = np.tensordot(Deltaeps,\
                                        np.tensordot(bigC, deltaE, axes = 2), axes = 2)
                            
                            
                            # Kmat_F = Kmat_F_1 + Kmat_F_2



                            # Kmat_F = np.dot(Deltaeps_voigt,np.dot(C_ela_2d_voigt,deltaE_voigt.transpose()).transpose())                             
                            # Kmat_el_2 = np.dot(S_all_voigt.transpose(),Delta_delta_E_voigt.transpose()) 




                            # Kmat_voigt = np.dot(C_ela_2d_voigt, Deltaeps_voigt)

                            # First terms of eos part
                            # F_e_inv=np.linalg.inv(F_e_current)
                            # DC_e=np.dot(gradX_Du.transpose(),F_e_current) + np.dot(F_e_current.transpose(),gradX_Du) # This is the same Delta delta E

                            # eos_1 = J*np.tensordot(F_e_inv.transpose(),gradX_Du,axes=2)*C_e_inv
                            # eos_2 = J*np.tensordot(dyad_bar,DC_e) # Using Delta delta E instead
                            # eos_2 = J*np.tensordot(dyad_bar,Delta_delta_E_voigt)
                            # Kmat_eos_1 = -p_eos*np.tensordot(eos_1+eos_2,deltaE,axes=2)
                            # Kmat_eos_2 = np.tensordot(S_eos,Delta_delta_E,axes=2)
                            # Kmat_eos = Kmat_eos_1 + Kmat_eos_2
                            # Kmat_eos= -p_eos *  (np.tensordot(J*F_e_inv.transpose(),gradX_Du,axes=2)) + np.tensordot(dyad_bar,DC_e,axes=2)

                            
                            # add to the corresponding entry in Ke and dont forget other parts of integral

                        
                            # Ke[ni*2+ci,nj*2+cj] += wi*np.linalg.det(dXdxi)*(Kgeom+Kmat)
                            # assemble into global 
                            KK[node_ei[ni]*2+ci,node_ei[nj]*2+cj] += wi*np.linalg.det(dXdxi)*(Kgeom+Kmat)
                            
    return RR,KK,F_p_next,g_next,S_next_all,F_e_next,F_next


def specialDyadUp(A,B):

    dyad_bar = np.zeros([2,2,2,2])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    dyad_bar[i,j,k,l] = A[i,k]*B[j,l] 

    

    return dyad_bar

def specialDyadDown(A,B):

    dyad_bar = np.zeros([2,2,2,2])
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    dyad_bar[i,j,k,l] = A[i,l]* B[j,k] 

    

    return dyad_bar