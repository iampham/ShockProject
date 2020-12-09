import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

######### READ ME ##########
# To plot notice there are 2 different times: t_plot_0 and t_plot
# you can select different times to see changes in 2 steps
# be careful with the polygons and with the plot/variables labels in the final plots when making final ones

############################


# data = np.load('FEAData.npz')
# data = np.load('./Data/FEAData_9_0_36_0_0deg.npz') # 2 Different
# data = np.load('./Data/FEAData_0_9_18_27_36deg.npz') # All different
data = np.load('./Data/FEAData_compression_50steps.npz')


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

# Plot Mesh, Stress and Configurations at desired time: tplot
# Select t_plot:
t_plot_0 = 0. # [s]
t_plot_ind_0 = int(t_plot_0/deltat)
t_plot = 0.5 # [s]
t_plot_ind = int(t_plot/deltat) # Time index for desired time

# TODO For reverse deformation process: plot t=0.1 and then regular timend

################################################

# Calculate average of each principal stress for each element and plot respective stress

# Get deformed nodes at desired time
node_x_plot = u_vec[:,:,t_plot_ind]

node_x_plot_0 = u_vec[:,:,t_plot_ind_0]

# Initialize plot stuff
fig, [[ax1,ax2,ax3],[ax4,ax5,ax6]] = plt.subplots(2,3,figsize=(15,9))
# TODO uncomment if needed
# ax1.set_xlim([-0.1, 1.2])
# ax1.set_ylim([-0.1, 1.2])
# ax1.set_aspect('equal')
# ax2.set_xlim([-0.1, 1.2])
# ax2.set_ylim([-0.1, 1.2])
# ax2.set_aspect('equal')
# ax3.set_xlim([-0.1, 1.2])
# ax3.set_ylim([-0.1, 1.2])
# ax3.set_aspect('equal')
# ax4.set_xlim([-0.1, 1.2])
# ax4.set_ylim([-0.1, 1.2])
# ax4.set_aspect('equal')
# ax5.set_xlim([-0.1, 1.2])
# ax5.set_ylim([-0.1, 1.2])
# ax5.set_aspect('equal')
# ax6.set_xlim([-0.1, 1.2])
# ax6.set_ylim([-0.1, 1.2])
# ax6.set_aspect('equal')
patches1 = []
patches2 = []
colors_sig11 = np.zeros([n_elem])
colors_sig22 = np.zeros([n_elem])
colors_sig12 = np.zeros([n_elem])
colors_sig11_0 = np.zeros([n_elem])
colors_sig22_0 = np.zeros([n_elem])
colors_sig12_0 = np.zeros([n_elem])
colors_S11 = np.zeros([n_elem])
colors_S22 = np.zeros([n_elem])
colors_S12 = np.zeros([n_elem])
colors_g_1 = np.zeros([n_elem])

# Get stress at time start
# Access data from each element
for ne in range(n_elem):
    n1 = elements[ne,0]
    n2 = elements[ne,1]
    n3 = elements[ne,2]
    n4 = elements[ne,3]
    polygon = Polygon([node_X[n1],node_X[n2],node_X[n3],node_X[n4]], closed=True) # Uncomment if you want to plot reference config
    # polygon = Polygon([node_x_plot_0[n1],node_x_plot_0[n2],node_x_plot_0[n3],node_x_plot_0[n4]], closed=True)
    patches1.append(polygon)
    polygon = Polygon([node_x_plot[n1],node_x_plot[n2],node_x_plot[n3],node_x_plot[n4]], closed=True)
    patches2.append(polygon)
    
    # Initialize principal values for each integration point
    sigma11 = np.zeros([n_IP])
    sigma22 = np.zeros([n_IP])
    sigma12 = np.zeros([n_IP]) 
    sigma11_0 = np.zeros([n_IP])
    sigma22_0 = np.zeros([n_IP])
    sigma12_0 = np.zeros([n_IP]) 
    S11 = np.zeros([n_IP])
    S22 = np.zeros([n_IP])
    S12 = np.zeros([n_IP]) 
    g_1 = np.zeros([n_IP]) 

    for ip in range(n_IP):        
        # Access each integration point for an element
        sigma_ip = np.zeros([2,2])
        sigma_ip = sigma_all[:,:,ne,ip,t_plot_ind]
        sigma11[ip] = sigma_ip[0,0]
        sigma22[ip] = sigma_ip[1,1]
        sigma12[ip] = sigma_ip[0,1]
        # Stress at start
        sigma_ip_0 = np.zeros([2,2])
        sigma_ip_0 = sigma_all[:,:,ne,ip,t_plot_ind_0]
        sigma11_0[ip] = sigma_ip_0[0,0]
        sigma22_0[ip] = sigma_ip_0[1,1]
        sigma12_0[ip] = sigma_ip_0[0,1]
        S_ip = np.zeros([2,2])
        S_ip = S_all[:,:,ne,ip,t_plot_ind]
        S11[ip] = S_ip[0,0]
        S22[ip] = S_ip[1,1]
        S12[ip] = S_ip[0,1]
        # g
        g_ip = np.zeros([10,1])
        g_ip = g_all[:,:,ne,ip,t_plot_ind]
        g_1[ip] = g_ip[0,0]

    # Store average of principal stresses for each element
    # Sigma colors avg for each element
    colors_sig11[ne] = np.mean(sigma11)
    colors_sig22[ne] = np.mean(sigma22)
    colors_sig12[ne] = np.mean(sigma12)
    colors_sig11_0[ne] = np.mean(sigma11_0)
    colors_sig22_0[ne] = np.mean(sigma22_0)
    colors_sig12_0[ne] = np.mean(sigma12_0)
    # S colors avg for each element
    colors_S11[ne] = np.mean(S11)
    colors_S22[ne] = np.mean(S22)
    colors_S12[ne] = np.mean(S12)
    # g
    colors_g_1[ne] = np.mean(g_1)


# Plot formatting
p1 = PatchCollection(patches1)
# p1 = PatchCollection(patches2)
# p1.set_array(colors_S11) # Second Piola
p1.set_array(colors_sig11_0) # Paint sigma stress
# p1.set_array(colors_g_1) # Paint with g
ax1.add_collection(p1)
p1.set_clim([102., 105.])
plt.colorbar(p1,ax=ax1,shrink=0.7)
ax1.set_title(("Reference Configuration at $t=%1.2f$ [s]\
\nCauchy Stress $\sigma_{11}$ at Center"%t_plot_0))


p2 = PatchCollection(patches1)
# p2 = PatchCollection(patches2)
# p2.set_array(colors_S22) # Second Piola
p2.set_array(colors_sig22_0)
ax2.add_collection(p2)
# p2.set_clim([0.18, 0.2])
plt.colorbar(p2,ax=ax2,shrink=0.7)
ax2.set_title(("Reference Configuration at $t=%1.2f$ [s]\
\nCauchy Stress $\sigma_{22}$ at Center"%t_plot_0))

p3 = PatchCollection(patches1)
# p3 = PatchCollection(patches2)
# p3.set_array(colors_S12) # Second Piola
p3.set_array(colors_sig12_0)
ax3.add_collection(p3)
# p3.set_clim([0., 0.2])
plt.colorbar(p3,ax=ax3,shrink=0.7)
ax3.set_title(("Reference Configuration at $t=%1.2f$ [s]\
\nCauchy Stress $\sigma_{12}$ at Center"%t_plot_0))

p4 = PatchCollection(patches2)
p4.set_array(colors_sig11)
ax4.add_collection(p4)
# p4.set_clim([0.18, 0.2])
plt.colorbar(p4,ax=ax4,shrink=0.7)
ax4.set_title("Deformed Configuration at $t=%1.2f$ [s]\
\nCauchy Stress $\sigma_{11}$ at Center"%t_plot)
ax4.scatter(node_x_plot_0[:,0],node_x_plot_0[:,1])

p5 = PatchCollection(patches2)
p5.set_array(colors_sig22)
ax5.add_collection(p5)
# p5.set_clim([0.18, 0.2])
plt.colorbar(p5,ax=ax5,shrink=0.7)
ax5.set_title("Deformed Configuration at $t=%1.2f$ [s]\
\nCauchy Stress $\sigma_{22}$ at Center"%t_plot)
ax5.scatter(node_x_plot_0[:,0],node_x_plot_0[:,1])

p6 = PatchCollection(patches2)
p6.set_array(colors_sig12)
ax6.add_collection(p6)
# p6.set_clim([0., 0.2])
plt.colorbar(p6,ax=ax6,shrink=0.7)
ax6.set_title("Deformed Configuration at $t=%1.2f$ [s]\
\nCauchy Stress $\sigma_{12}$ at Center"%t_plot)
ax6.scatter(node_x_plot_0[:,0],node_x_plot_0[:,1])


# print(sigma22_0)
print(g_1)

fig.tight_layout()
plt.show()