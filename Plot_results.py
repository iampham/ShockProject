import numpy as np
import matplotlib.pyplot as plt

data = np.load('FEAData.npz')

# Access the variables stored within the file:
S_all = data['S_all']
F_all = data['F_all']
F_e_all = data['F_e_all']
F_p_all = data['F_p_all']
g_all = data['g_all']
timeStart = data['timeStart']
timeEnd = data['timeEnd']
nSteps = data['nSteps']
n_IP = data['n_IP']
n_elem = data['n_elem']

################################################
########### Choose variables to plot ###########

# Time from Global Outline
t_vec = np.linspace(timeStart,timeEnd, nSteps)

# Element to get data from (Choosing to plot center element)
elem_plot = 4 # Choose elemnt to plot from n_elem

################################################

# Initialize values accross time
detF_t = np.zeros([nSteps])
detFe_t = np.zeros([nSteps])
detFp_t = np.zeros([nSteps])

# Access data from each time step
for t in range(nSteps):
    # Initialize values at each time step
    detF = np.zeros([n_IP])
    detFe = np.zeros([n_IP])
    detFp = np.zeros([n_IP])    
    # Access each integration point of a given element number
    for i in range(n_IP):
        detF[i] = np.linalg.det(F_all[:,:,elem_plot,i,t])
        detFe[i] = np.linalg.det(F_e_all[:,:,elem_plot,i,t])
        detFp[i] = np.linalg.det(F_p_all[:,:,elem_plot,i,t])

    # Take average of all the integration points
    detF_t[t] = np.mean(detF)
    detFe_t[t] = np.mean(detFe)
    detFp_t[t] = np.mean(detFp)

plt.plot(t_vec ,detF_t,"x",label="detF")
plt.plot(t_vec ,detFe_t,label="detFe")
plt.plot(t_vec ,detFp_t,label="detFp")
plt.legend()
plt.show()