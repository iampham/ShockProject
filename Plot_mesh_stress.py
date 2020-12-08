import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

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

# Plot Mesh, Stress and Configurations at desired time: tplot
# Select t_plot:
t_plot = 0.2 # [s]
t_plot_ind = int(t_plot/deltat) # Time index for desired time

################################################

# Calculate average of each principal stress for each element and plot respective stress

# Get deformed nodes at desired time
node_x_plot = u_vec[:,:,t_plot_ind]

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
colors_S11 = np.zeros([n_elem])
colors_S22 = np.zeros([n_elem])
colors_S12 = np.zeros([n_elem])

# Get stress at desired time
# Access data from each element
for ne in range(n_elem):
    n1 = elements[ne,0]
    n2 = elements[ne,1]
    n3 = elements[ne,2]
    n4 = elements[ne,3]
    polygon = Polygon([node_X[n1],node_X[n2],node_X[n3],node_X[n4]], closed=True)
    patches1.append(polygon)
    polygon = Polygon([node_x_plot[n1],node_x_plot[n2],node_x_plot[n3],node_x_plot[n4]], closed=True)
    patches2.append(polygon)
    
    # Initialize principal values for each integration point
    sigma11 = np.zeros([n_IP])
    sigma22 = np.zeros([n_IP])
    sigma12 = np.zeros([n_IP]) 
    S11 = np.zeros([n_IP])
    S22 = np.zeros([n_IP])
    S12 = np.zeros([n_IP]) 

    for ip in range(n_IP):        
        # Access each integration point for an element
        sigma_ip = np.zeros([2,2])
        sigma_ip = sigma_all[:,:,ne,ip,t_plot_ind]
        sigma11[ip] = sigma_ip[0,0]
        sigma22[ip] = sigma_ip[1,1]
        sigma12[ip] = sigma_ip[0,1]
        S_ip = np.zeros([2,2])
        S_ip = S_all[:,:,ne,ip,t_plot_ind]
        S11[ip] = S_ip[0,0]
        S22[ip] = S_ip[1,1]
        S12[ip] = S_ip[0,1]

    # Store average of principal stresses for each element
    # Sigma colors avg for each element
    colors_sig11[ne] = np.mean(sigma11)
    colors_sig22[ne] = np.mean(sigma22)
    colors_sig12[ne] = np.mean(sigma12)
    # S colors avg for each element
    colors_S11[ne] = np.mean(S11)
    colors_S22[ne] = np.mean(S22)
    colors_S12[ne] = np.mean(S12)

# Plot formatting
p1 = PatchCollection(patches1)
p1.set_array(colors_S11)
ax1.add_collection(p1)
# p1.set_clim([0.18, 0.2])
plt.colorbar(p1,ax=ax1,shrink=0.7)
ax1.set_title(("Reference Configuration at $t=0$ [s]\
\nSecond Piola-Kirchoff $S_{11}$ at Center"))

p2 = PatchCollection(patches1)
p2.set_array(colors_S22)
ax2.add_collection(p2)
# p2.set_clim([0.18, 0.2])
plt.colorbar(p2,ax=ax2,shrink=0.7)
ax2.set_title(("Reference Configuration at $t=0$ [s]\
\nSecond Piola-Kirchoff Stress $S_{22}$ at Center"))

p3 = PatchCollection(patches1)
p3.set_array(colors_S12)
ax3.add_collection(p3)
# p3.set_clim([0., 0.2])
plt.colorbar(p3,ax=ax3,shrink=0.7)
ax3.set_title(("Reference Configuration at $t=0$ [s]\
\nSecond Piola-Kirchoff Stress $S_{12}$ at Center"))

p4 = PatchCollection(patches2)
p4.set_array(colors_sig11)
ax4.add_collection(p4)
# p4.set_clim([0.18, 0.2])
plt.colorbar(p4,ax=ax4,shrink=0.7)
ax4.set_title("Deformed Configuration at $t=%1.1f$ [s]\
\nCauchy Stress $\sigma_{11}$ at Center"%t_plot)

p5 = PatchCollection(patches2)
p5.set_array(colors_sig22)
ax5.add_collection(p5)
# p5.set_clim([0.18, 0.2])
plt.colorbar(p5,ax=ax5,shrink=0.7)
ax5.set_title("Deformed Configuration at $t=%1.1f$ [s]\
\nCauchy Stress $\sigma_{22}$ at Center"%t_plot)

p6 = PatchCollection(patches2)
p6.set_array(colors_sig12)
ax6.add_collection(p6)
# p6.set_clim([0., 0.2])
plt.colorbar(p6,ax=ax6,shrink=0.7)
ax6.set_title("Deformed Configuration at $t=%1.1f$ [s]\
\nCauchy Stress $\sigma_{12}$ at Center"%t_plot)

fig.tight_layout()
plt.show()