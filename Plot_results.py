import numpy as np
import matplotlib.pyplot as plt

data = np.load('FEAData.npz')

# Access the variables stored within the file:
S_all = data['S_all']
sigma_all = data['sigma_all']
F_all = data['F_all']
F_e_all = data['F_e_all']
F_p_all = data['F_p_all']
g_all = data['g_all']
timeStart = data['timeStart']
timeEnd = data['timeEnd']
nSteps = data['nSteps']
n_IP = data['n_IP']
n_elem = data['n_elem']
u_vec = data['u_vec']
node_X = data['node_X']
elements = data['elements']

################################################
########### Choose variables to plot ###########

# Time from Global Outline
t_vec = np.linspace(timeStart,timeEnd, nSteps)
deltat = t_vec[1] - t_vec[0]

# Element to get data from (Choosing to plot center element)
elem_plot = 4 # Choose elemnt to plot from n_elem

################################################

# Plot variables over time for a given element

# Initialize plot stuff
fig, [ax1,ax2,ax3,ax4] = plt.subplots(1,4,figsize=(17,4))

# Initialize values accross time
detF_t = np.zeros([nSteps])
detFe_t = np.zeros([nSteps])
detFp_t = np.zeros([nSteps])
sig11_t = np.zeros([nSteps])
sig22_t = np.zeros([nSteps])
sig12_t = np.zeros([nSteps])

# Access data from each time step
for t in range(nSteps):
    # Initialize values at each time step
    detF = np.zeros([n_IP])
    detFe = np.zeros([n_IP])
    detFp = np.zeros([n_IP]) 
    sigma11 = np.zeros([n_IP])
    sigma22 = np.zeros([n_IP])
    sigma12 = np.zeros([n_IP])    
    # Access each integration point of a given element number
    for i in range(n_IP):
        # Deformation gradients
        detF[i] = np.linalg.det(F_all[:,:,elem_plot,i,t])
        detFe[i] = np.linalg.det(F_e_all[:,:,elem_plot,i,t])
        detFp[i] = np.linalg.det(F_p_all[:,:,elem_plot,i,t]) # TODO need to fix Fp sync issue (F_p is actually storing values at the next time step - Same as g)
        # Cauchy stress
        sigma_ip = np.zeros([2,2])
        sigma_ip = sigma_all[:,:,elem_plot,i,t]
        sigma11[i] = sigma_ip[0,0]
        sigma22[i] = sigma_ip[1,1]
        sigma12[i] = sigma_ip[0,1]

    # Take average of all the integration points
    detF_t[t] = np.mean(detF)
    detFe_t[t] = np.mean(detFe)
    detFp_t[t] = np.mean(detFp)
    sig11_t[t] = np.mean(sigma11)
    sig22_t[t] = np.mean(sigma22)
    sig12_t[t] = np.mean(sigma12)

ax1.plot(t_vec ,detF_t,"x",label="detF")
ax1.plot(t_vec ,detFe_t,label="detFe")
ax1.plot(t_vec ,detFp_t,label="detFp")
ax1.set_title('Determinant of deformation\nElement average (%1.0f) average'%elem_plot)
ax1.set_xlabel('$t$ [s]')
ax1.set_ylabel('$det(F)$')
ax1.legend()

ax2.plot(t_vec ,sig11_t)
ax2.set_title('Principal Cauchy Stress $\sigma_{11}$\nElement (%1.0f) average'%elem_plot)
ax2.set_xlabel('$t$ [s]')
ax2.set_ylabel('$\sigma_{11}$')
# ax2.legend()

ax3.plot(t_vec ,sig22_t)
ax3.set_title('Principal Cauchy Stress $\sigma_{22}$\nElement (%1.0f) average'%elem_plot)
ax3.set_xlabel('$t$ [s]')
ax3.set_ylabel('$\sigma_{22}$')
# ax3.legend()

ax4.plot(t_vec ,sig12_t)
ax4.set_title('Principal Cauchy Stress $\sigma_{12}$\nElement (%1.0f) average'%elem_plot)
ax4.set_xlabel('$t$ [s]')
ax4.set_ylabel('$\sigma_{12}$')
# ax4.legend()

fig.tight_layout()
plt.show()