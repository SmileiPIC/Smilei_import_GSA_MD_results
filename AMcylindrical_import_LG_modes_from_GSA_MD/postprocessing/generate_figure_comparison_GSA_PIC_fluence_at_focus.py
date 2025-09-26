###### Authors: M. Masckala, F. Massimo
 
###### Comparison between the GSA-MD-reconstructed fluence at focus and the one simulated with a PIC code

from gsa_md.mode_basis.laguerre_gauss_modes import *

import happi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

############### Load the results of the GSA-MD laser field reconstruction ######
dict_image_preprocessing = np.load('./outputs_cylindrical_GSA_MD/dict_image_preprocessing.npy',allow_pickle=True).item()
dict_mesh                = np.load('./outputs_cylindrical_GSA_MD/dict_mesh.npy',allow_pickle=True).item()
dict_mode_basis          = np.load('./outputs_cylindrical_GSA_MD/dict_mode_basis.npy',allow_pickle=True).item()
Coeffs_LG_pl             = np.load('./outputs_cylindrical_GSA_MD/Coeffs_MD_iteration_00199.npy')


######################## Open the results from Smilei ##########################
S                        = happi.Open()

## Physical constants from namelist
lambda_0                 = S.namelist.lambda_0      # laser wavelength, m
x_focus_meter            = S.namelist.x_focus_LG_modes_GSA_MD 
c_normalized             = S.namelist.c_normalized  # speed of light in vacuum in normalized units
um                       = S.namelist.um            # 1 micron in normalized units
meter                    = S.namelist.meter         # 1 meter in normalized units
dt                       = S.namelist.dt
Lx                       = S.namelist.Lx 
#distance_x_focus_um      = S.nam#S.namelist.distance_x_focus_um

## Extract the mesh from the probe field (Ey) axis (y,z)
probe                    = S.Probe.Probe3("Ey") 
y_axis                   = probe.getAxis("axis1")
z_axis                   = probe.getAxis("axis2")

## Evaluate the size of the mesh axis
ny                       = y_axis.shape[0]
nz                       = z_axis.shape[0]

# choose only the y coordinates from y_axis and the z coordinates from z_axis
y    = np.array([arr[1] for arr in y_axis])/(1.e6*um)
z    = np.array([arr[2] for arr in z_axis])/(1.e6*um)

# Y[i,j], Z[i,j] give respectively the y and z coordinates of the point [i.j] in the mesh
Y, Z = np.meshgrid(y, z, indexing="ij")

# Definition of the radial axis r and theta axis (each has the same shape as Y and Z)
r_mesh_for_interpolation = np.sqrt(np.square(Y)+np.square(Z))        # meters
theta_mesh               = np.arctan2(Z,Y)

# Flatten r_mehs and theta to be used in LG_pl_field_x_r
r_mesh_for_interpolation = r_mesh_for_interpolation.flatten()
theta_mesh               = theta_mesh.flatten()

################### Calculation of Ey_complex and Fluence ######################

# Store the sum_p LG^{pl} field for each l value
# For this, define an interpolation r_mesh, the focal plane and then multiply each mode by 
# its coefficient found by the field reconstruction algorithm

dict_mesh_plane_xmin     = {"plane_x_coordinates":np.array([0]),"r_mesh":r_mesh_for_interpolation, "theta_mesh": theta_mesh} 

Ey_complex               = np.zeros(shape=(1, np.size(r_mesh_for_interpolation)),dtype=complex)
LG_l_field_r             = np.zeros(shape=(2*dict_mode_basis["Max_LG_index_l"]+1,1,np.size(r_mesh_for_interpolation)),dtype=complex)
for l in range(-dict_mode_basis["Max_LG_index_l"], dict_mode_basis["Max_LG_index_l"] + 1):
    # negative l indices are stored with the FFT convention for negative frequencies
    l_index = l if l >= 0 else dict_mode_basis["Max_LG_index_l"] + abs(l)
    # sum the x,r part of the modes with the same l index and different radial p indices
    for p in range(0,dict_mode_basis["Max_LG_index_p"]+1):
        LG_l_field_r[l_index,:,:] += Coeffs_LG_pl[p,l_index]  *LG_pl_field_x_r(lambda_0,p,l,dict_mesh_plane_xmin,dict_mode_basis ,check_total_power_integral=False)
    # modes with the same l share the same exp(i*l*theta)
    LG_l_field_r[l_index,:,:]      = LG_l_field_r[l_index,:,:]*LG_pl_field_theta(l,dict_mesh_plane_xmin,dict_mode_basis,check_total_power_integral=False)
    # add to the total field
    Ey_complex += LG_l_field_r[l_index,:,:]

Ey_complex      = Ey_complex.reshape(ny, nz)

## Fluence calculated based on GSA_MD extracted coefficients at x_focus
E_sq_cal        = np.square(np.abs(Ey_complex))

## Extract the field at timestep_focus or around it. Start from timestep_focus as a reference  
timestep_focus  = np.round(S.namelist.time_laser_at_LG_x_focus/dt)-2

## Fluence from simulation at x_focus
probe           = S.Probe.Probe3("Ey**2", units=['um'])
E_sq_sim        = np.array(probe.getData(timestep=timestep_focus))
E_sq_sim        = E_sq_sim.reshape(ny, nz)

##########################  Comparison Plot  ###################################

# Define the colors for the colormap
colors = [(1.        , 1.        , 1.       ),   # white
          (0.        , 0.21875   , 0.65625  ),   # blue
          (0.60546875, 0.30859375, 0.5859375),   # purple
          (0.8359375 , 0.0078125 , 0.4375   )]   # pink

# Create the colormap
cmap_name                    = 'my_cmap'
my_cmap                      = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors, N=256)

# Extract the maximum fluence from Exp results 
dict_image_preprocessing     = np.load('outputs_cylindrical_GSA_MD/dict_image_preprocessing.npy',allow_pickle=True).item()
fluence_exp_images_cartesian = dict_image_preprocessing["fluence_exp_images_cartesian"]
vmax                         = 0.7*np.amax(fluence_exp_images_cartesian)

# Set up the subplots
fig, axs = plt.subplots(1, 3, figsize=(12, 5))

# Plot I_cal
im0 = axs[0].imshow(E_sq_cal, origin='lower', extent=[min(z*1e6), max(z*1e6), min(y*1e6), max(y*1e6)],
                    aspect='equal', cmap=my_cmap, vmin=0, vmax=vmax)
axs[0].set_title(r"$\|\tilde{E}_{cal}\|^2 ~[arb. units]$")
axs[0].set_xlabel("z (μm)")
axs[0].set_ylabel("y (μm)")

# Plot I_sim
im1 = axs[1].imshow(E_sq_sim, origin='lower', extent=[min(z*1e6), max(z*1e6), min(y*1e6), max(y*1e6)],
                    aspect='equal', cmap=my_cmap, vmin=0, vmax=vmax)
axs[1].set_title(r"$\|\tilde{E}_{sim}\|^2 ~[arb. units]$")
axs[1].set_xlabel("z (μm)")
# axs[1].set_ylabel("y (μm)")


## Plot the Absolute error (|E|^2_{cal} - |E|^2_{sim} )
delta_E_sq = np.abs(E_sq_cal-E_sq_sim)
im1 = axs[2].imshow(delta_E_sq, origin='lower', extent=[min(z*1e6), max(z*1e6), min(y*1e6), max(y*1e6)],
                    aspect='equal', cmap=my_cmap, vmin=0, vmax=vmax)
axs[2].set_title(r"$| \|\tilde{E}_{cal}\|^2 - \|\tilde{E}_{sim}\|^2| ~[arb. units]$")
axs[2].set_xlabel("z (μm)")
# axs[2].set_ylabel("y (μm)")

# Add a shared colorbar
cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])
fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', label='|E|^2')

# Print the relative absolute error
err_abs_percent = np.round((delta_E_sq.max())/np.amax(E_sq_cal)*100, 3)
print(f"The maximum absolute relative error is {err_abs_percent} %")

# Save the figure
plt.show()
plt.savefig("Comparison_GSA_MD_Smilei_at_focus.pdf",format="pdf",dpi=1000)

