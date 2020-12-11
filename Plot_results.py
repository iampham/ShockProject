import numpy as np
import matplotlib.pyplot as plt


######### READ ME ##########
# To plot notice there are 2 data sets: data and data1
# make sure to have the appropiate time steps for each one if you are going to plot both

# Also select the element that you want to see any trends over time (default is 4=center)

############################


# data = np.load('FEAData.npz')
data = np.load('./Data/FEAData_expansion.npz')

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

data1 = np.load('./Data/FEAData_expansion.npz')

# Access the variables stored within the file:
# S_all_1 = data1['S_all']
sigma_all_1 = data1['sigma_all']
# F_all_1 = data1['F_all']
# F_e_all_1 = data1['F_e_all']
# F_p_all_1 = data1['F_p_all']
# g_all_1= data1['g_all']
timeStart_1 = data1['timeStart']
timeEnd_1 = data1['timeEnd']
nSteps_1 = data1['nSteps']
# n_IP_1 = data1['n_IP']
# n_elem_1 = data1['n_elem']
# u_vec_1 = data1['u_vec']
# node_X_1 = data1['node_X']
# elements_1 = data1['elements']

################################################
########### Choose variables to plot ###########

# Time from Global Outline
t_vec = np.linspace(timeStart,timeEnd, nSteps)
deltat = t_vec[1] - t_vec[0]

t_vec_1 = np.linspace(timeStart_1,timeEnd_1, nSteps_1)

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
sig11_t_1 = np.zeros([nSteps])
sig22_t_1 = np.zeros([nSteps])
sig12_t_1 = np.zeros([nSteps])

# Access data from each time step
for t in range(nSteps):
    # Initialize values at each time step
    detF = np.zeros([n_IP])
    detFe = np.zeros([n_IP])
    detFp = np.zeros([n_IP]) 
    sigma11 = np.zeros([n_IP])
    sigma22 = np.zeros([n_IP])
    sigma12 = np.zeros([n_IP])    
    sigma11_1 = np.zeros([n_IP])
    sigma22_1 = np.zeros([n_IP])
    sigma12_1 = np.zeros([n_IP]) 
    # Access each integration point of a given element number
    for i in range(n_IP):
        # Deformation gradients
        detF[i] = np.linalg.det(F_all[:,:,elem_plot,i,t])
        detFe[i] = np.linalg.det(F_e_all[:,:,elem_plot,i,t])
        detFp[i] = np.linalg.norm(F_p_all[:,:,elem_plot,i,t]) # TODO need to fix Fp sync issue (F_p is actually storing values at the next time step - Same as g)
        # Cauchy stress
        sigma_ip = np.zeros([2,2])
        sigma_ip = sigma_all[:,:,elem_plot,i,t]
        sigma11[i] = sigma_ip[0,0]
        sigma22[i] = sigma_ip[1,1]
        sigma12[i] = sigma_ip[0,1]
        # Second data set
        sigma_ip_1 = np.zeros([2,2])
        sigma_ip_1 = sigma_all_1[:,:,elem_plot,i,t]
        sigma11_1[i] = sigma_ip_1[0,0]
        sigma22_1[i] = sigma_ip_1[1,1]
        sigma12_1[i] = sigma_ip_1[0,1]

    # Take average of all the integration points
    detF_t[t] = np.mean(detF)
    detFe_t[t] = np.mean(detFe)
    detFp_t[t] = np.mean(detFp)
    sig11_t[t] = np.mean(sigma11)
    sig22_t[t] = np.mean(sigma22)
    sig12_t[t] = np.mean(sigma12)
    sig11_t_1[t] = np.mean(sigma11_1)
    sig22_t_1[t] = np.mean(sigma22_1)
    sig12_t_1[t] = np.mean(sigma12_1)

# ax1.plot(t_vec ,detF_t,"x",label="detF")
# ax1.plot(t_vec ,detFe_t,label="detFe")
# ax1.plot(t_vec ,detFp_t,label="detFp")
ax1.plot(t_vec ,detFp_t,label="norm(Fp)")
ax1.set_title('Determinant of deformation\nElement average (%1.0f) average'%elem_plot)
ax1.set_xlabel('$t$ [s]')
ax1.set_ylabel('$det(F)$')
ax1.legend()

ax2.plot(t_vec ,sig11_t,label="2 different")
ax2.plot(t_vec ,sig11_t_1,label="all different")
# ax2.plot(t_vec_1 ,sig11_t_1,label="all different")
ax2.set_title('Principal Cauchy Stress $\sigma_{11}$\nElement (%1.0f) average'%elem_plot)
ax2.set_xlabel('$t$ [s]')
ax2.set_ylabel('$\sigma_{11}$')
ax2.legend()

ax3.plot(t_vec ,sig22_t,label="2 different")
ax3.plot(t_vec ,sig22_t_1,label="all different")
# ax3.plot(t_vec_1 ,sig22_t_1,label="all different")
ax3.set_title('Principal Cauchy Stress $\sigma_{22}$\nElement (%1.0f) average'%elem_plot)
ax3.set_xlabel('$t$ [s]')
ax3.set_ylabel('$\sigma_{22}$')
ax3.legend()

ax4.plot(t_vec ,sig12_t,label="2 different")
ax4.plot(t_vec ,sig12_t_1,label="all different")
# ax4.plot(t_vec_1 ,sig12_t_1,label="all different")
ax4.set_title('Principal Cauchy Stress $\sigma_{12}$\nElement (%1.0f) average'%elem_plot)
ax4.set_xlabel('$t$ [s]')
ax4.set_ylabel('$\sigma_{12}$')
ax4.legend()

fig.tight_layout()
plt.show()