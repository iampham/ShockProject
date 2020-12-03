import numpy as np 
from calculatePlastic import * 
from computeSecondPiola import *

def assembleRRKK(const_dictionary,Nvec, dNvecdxi, n_nodes, n_elem, elements, node_X, node_x,F_p_prev,g_prev,dt,shock_bound):
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
    C_elastic = const_dictionary["C_ela"]
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
    S_next=np.zeros([2,2,n_elem,n_IP])
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
            # 2 - Need values for internal functions

            # Slip resistance stop parameter We don't actually use this
            g_tol = 1000

            g_itermax = 1
            g_iter = 0
            g_not_converged = True
            g_prev_loc=g_prev[:,:,ei,ip]

            while (g_not_converged and g_iter<g_itermax):
                
                # Pre solve stress
                # according to the current shockwave boundary and coordinates of ip, update v
                if x[0]<shock_bound:
                    S_prev,S_eos,S_el_voigt,p_eos=computeSecondPiola(F_e_prev_loc,const_dictionary,v)
                else:
                    S_prev,S_eos,S_el_voigt,p_eos=computeSecondPiola(F_e_prev_loc,const_dictionary,v_0)

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

                    # compute delta_S and add it to S
                    delta_S = -np.tensordot(np.linalg.inv(J_S),res_S,axes=2)
                    res_S=res_S[0:2,0:2]
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

                    
                F_p_current, g_current = calculateNextPlastic(F_p_prev_loc,gamma_dot_ref, m, g_sat, g_prev_loc, a, h, dt, F_e_current, S_current)

                g_diff = np.linalg.abs(g_prev_loc-g_current)
                if g_diff>g_max:
                    g_max = g_diff


                g_not_converged = notConvergedYet(g_prev_loc,g_current,g_tol)
                g_iter += 1

                # TODO we are getting a huge F_p (1e80+) and that makes F_e really small
                # print("F_p_prev_loc",F_p_prev_loc,"gamma_dot_ref",gamma_dot_ref,"m",m, "g_sat",g_sat, "g_prev_loc",g_prev_loc,\
                #       "a",a, "h",h, "dt",dt, "F_e_prev_loc",F_e_prev_loc, "S_prev",S_prev)

                # # F_p = np.eye(2) # TODO 27 nov 2020 debugging proved that elastic part is working properly, still need to figure out how to make the plastic part work
                # F_e = np.dot(F,  np.linalg.inv(F_p))            
                # # print("F",F)
                # # print("F_p",F_p)
                # # print("F_e",F_e)



                # J = np.linalg.det(F_e)

                # C_e = np.dot(F_e.transpose(), F_e)
                # # special dyadic product of C_e
                # C_e_inv = np.linalg.inv(C_e)
                # # C_inv Special Dyad C_inv
                # dyad_bar = np.zeros([2,2,2,2])
                # for i in range(2):
                #     for j in range(2):
                #         for k in range(2):
                #             for l in range(2):
                #                 dyad_bar[i,j,k,l] = -C_e_inv[i,k]*C_e_inv[j,l] 

                # if x[0]<shock_bound:
                #     S,S_eos,S_el_voigt,p_eos=computeSecondPiola(F_e,const_dictionary,v)
                # else:
                #     S,S_eos,S_el_voigt,p_eos=computeSecondPiola(F_e,const_dictionary,v_0)       

    
            # F_p_current = np.dot(F,np.linalg.inv(F_p))            
            # F_p_prev_loc = F_p_prev[:,:,ei,ip]
            # calculate F_e from F and F_p
            # F_e_prev_loc = np.dot(F,np.linalg.inv(F_p_prev_loc)) 

            # Post Solve Stress

            # Compute Second Piola Kirchoff
            E = 0.5*((np.dot(F_e.transpose(),F_e)-np.eye(2))

            # Compute Cauchy with actual value of elastic deformation
            J = np.linalg.det(F)  
            sigma = (1/J)*np.dot(F_e_current,np.dot(S_current,F_e_current.transpose()))
            
            # store results in global variables
            g_next[:,:,ei,ip] = g_current
            sigma_next[:,:,ei,ip] = sigma
            S_next[:,:,ei,ip] = S_current
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
                    Re[ni*2+ci] += wi*np.linalg.det(dXdxi)*np.tensordot(S,deltaE)

                    # ASSEMBLE INTO GLOBAL RESIDUAL (I didn't ask for this)
                    RR[node_ei[ni]*2+ci] += wi*np.linalg.det(dXdxi)*np.tensordot(S,deltaE)
                    
                    ## 2 more for loops for the increment Delta u
                    for nj in range(4):
                        for cj in range(2):
                            Deltau = np.zeros((2))
                            Deltau[cj]=1
                            gradX_Du = np.outer(Deltau,dNsdX[:,nj])
                            Delta_delta_E = 0.5*np.dot(gradX_v.transpose(),gradX_Du)+\
                                np.dot(gradX_Du.transpose(),gradX_v)

                            Delta_delta_E_voigt = np.array([[Delta_delta_E[0,0],Delta_delta_E[1,1],2*Delta_delta_E[0,1]]])
                            
                            Deltaeps = 0.5*(np.dot(gradX_Du.transpose(),F_e) + np.dot(F_e.transpose(),gradX_Du))
                            
                            ## ELEMENT TANGENT
                            # Initial stress component (also called geometric component) is 
                            # refer to Linearization.pdf
                            Kgeom = np.tensordot(S,np.dot(gradX_v.transpose(),gradX_Du) + np.dot(gradX_Du.transpose(),gradX_v),axes=2)
                            # Material component, need to put things in voigt notation for easy computation
                            # deltad_voigt = np.array([deltad[0,0],deltad[1,1],2*deltad[0,1]])
                            
                            # D = np.array([[4*p,2*p,0],[2*p,4*p,0],[0,0,2*p]])
                            Deltaeps_voigt = np.array([Deltaeps[0,0],Deltaeps[1,1],2*Deltaeps[0,1]])
                            # First terms of elastic material part
                            Kmat_el_1 = np.dot(Deltaeps_voigt,np.dot(C_elastic,deltaE_voigt.transpose()).transpose())     

                                    
                            Kmat_el_2 = np.dot(S_el_voigt.transpose(),Delta_delta_E_voigt.transpose())
                            Kmat_el = Kmat_el_1 + Kmat_el_2
                            # Kmat_el_voigt = np.dot(C_elastic, Deltaeps_voigt)

                            # First terms of eos part
                            F_e_inv=np.linalg.inv(F_e)
                            DC_e=np.dot(gradX_Du.transpose(),F_e) + np.dot(F_e.transpose(),gradX_Du)

                            eos_1 = J*np.tensordot(F_e_inv.transpose(),gradX_Du,axes=2)*C_e_inv
                            eos_2 = J*np.tensordot(dyad_bar,DC_e)
                            Kmat_eos_1 = -p_eos*np.tensordot(eos_1+eos_2,deltaE,axes=2)
                            Kmat_eos_2 = np.tensordot(S_eos,Delta_delta_E,axes=2)
                            Kmat_eos = Kmat_eos_1 + Kmat_eos_2
                            # Kmat_eos= -p_eos *  (np.tensordot(J*F_e_inv.transpose(),gradX_Du,axes=2)) + np.tensordot(dyad_bar,DC_e,axes=2)

                            
                            Kmat = Kmat_el + Kmat_eos
                            # add to the corresponding entry in Ke and dont forget other parts of integral

                        
                            Ke[ni*2+ci,nj*2+cj] += wi*np.linalg.det(dXdxi)*(Kgeom+Kmat)
                            # assemble into global 
                            KK[node_ei[ni]*2+ci,node_ei[nj]*2+cj] += wi*np.linalg.det(dXdxi)*(Kgeom+Kmat)
                            
    return RR,KK,F_p_next,g_next,S_next,F_e_next,F_next