###### Authors: M. Masckala, F. Massimo

###### Comparison at focus between the GSA-MD-reconstructed width 
###### and the one extracted from a Smilei PIC simulation which uses the 
###### reconstructed field mode coefficients as input

import happi
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from numba import jit,prange,njit


##################### Plot the data from Smilei simulation #####################
S                 = happi.Open()
timesteps         = S.Probe.Probe1("Ey").getAvailableTimesteps()
timesteps         = timesteps[2::]

y_axis            = np.arange(int(2*100e-6/S.namelist.length_per_pixel_exp))*S.namelist.length_per_pixel_exp
y_axis            = y_axis-y_axis.max()/2.

z_axis            = y_axis

one_ov_e2_width_y = []
one_ov_e2_width_z = []

## Auxiliary functions
def E_field_to_Envelope_of_E(E_field):
    return hilbert(E_field, axis=0)

@njit(parallel=True)
def Env_E_field_abs(Env_E, Env_E_abs):
    for i in prange(Env_E.shape[0]):
        for j in range(Env_E.shape[1]):
            Env_E_abs[i, j] = np.abs(Env_E[i, j])
    return Env_E_abs

@njit(parallel=True)
def compute_fluence_along_axis(Env_E_abs, fluence):
    for j in prange(Env_E_abs.shape[1]):
        acc = 0.0
        for i in range(Env_E_abs.shape[0]):
            acc += Env_E_abs[i, j] * Env_E_abs[i, j]
        fluence[j] = acc
    return fluence

def compute_width_at_level_times_peak_value(x,y,level=0.5):
    
    ##### Given a curve y=f(x) with a peak y_peak, this function computes the 
    ##### width (in x units) of the peak, defined as the difference between the first x positions around the peak which
    ##### have y = level*y_peak.
    ##### when level = 0.5 you are computing the FWHM;
    ##### when level = 1 / math.e ** 2, you are computing the 1/e^2 width
    
    x_array_midpoints = (x[1:]+x[:-1])/2. # array containing the midpoints in the x array
    index_y_peak      = np.where(y==y.max())[0][0]
    x_peak            = x_array_midpoints[index_y_peak]
    y_peak            = y[index_y_peak]
    
    # Iterative determination of FWHM : from the x_peak,
    # the x where y = level*y_peak are searched, on the left and on the right
    #print("Peak at ", x_peak, ", max =",y_peak)
    left_peak_index   = index_y_peak
    right_peak_index  = index_y_peak
    while (y[left_peak_index]>y_peak*level and left_peak_index > 1):
        left_peak_index  = left_peak_index - 1 
    while (y[right_peak_index]>y_peak*level and right_peak_index < np.size(y)-2):
        right_peak_index = right_peak_index +1
        
    width             = x_array_midpoints[right_peak_index] - x_array_midpoints[left_peak_index]

    #print("Peak at ", x_peak, ", max y =",y_peak,", width ", width )
    #print("level edges = ",x_array_midpoints[left_peak_index],", ",x_array_midpoints[right_peak_index])
    
    # returns the left and right borders of the peak, the peak width and the position of the peak
    return width

### Extract data from the PIC simulation
### These loops can take time, 
### it is recommended to use multiple cpu cores 
### to leverage the numba parallelization
i=0
ntimesteps = len(timesteps)
for timestep in timesteps:
    i+=1
    print("y direction, reading timestep ",i," of ",ntimesteps)
    Ey_on_plane_xy      = S.Probe.Probe1("Ey",timesteps=timestep).getData()
    Ey_on_plane_xy      = np.asarray(Ey_on_plane_xy)[0,:,:]
    Env_Ey_plane_xy     = E_field_to_Envelope_of_E(Ey_on_plane_xy)
    Env_Ey_plane_xy_abs = np.zeros_like(Env_Ey_plane_xy,dtype=float)
    Env_Ey_plane_xy_abs = Env_E_field_abs(Env_Ey_plane_xy,Env_Ey_plane_xy_abs);#plt.figure();plt.imshow(Env_Ey_plane_xy_abs.T,aspect="auto",origin="lower")
    fluence_Ey_axis_y   = np.zeros_like(Env_Ey_plane_xy_abs[0,:])
    fluence_Ey_axis_y   = compute_fluence_along_axis(Env_Ey_plane_xy_abs,fluence_Ey_axis_y)
    try:
        two_times_waist_y = compute_width_at_level_times_peak_value(y_axis,fluence_Ey_axis_y,level=1./np.e**2)
        one_ov_e2_width_y.append(two_times_waist_y)
    except:
        one_ov_e2_width_y.append(0.)
    del Ey_on_plane_xy,Env_Ey_plane_xy,Env_Ey_plane_xy_abs,fluence_Ey_axis_y

i=0
for timestep in timesteps:
    i+=1
    print("z direction, reading timestep ",i," of ",ntimesteps)
    Ey_on_plane_xz      = S.Probe.Probe2("Ey",timesteps=timestep).getData()
    Ey_on_plane_xz      = np.asarray(Ey_on_plane_xz)[0,:,:]
    Env_Ey_plane_xz     = E_field_to_Envelope_of_E(Ey_on_plane_xz)
    Env_Ey_plane_xz_abs = np.zeros_like(Env_Ey_plane_xz,dtype=float)
    Env_Ey_plane_xz_abs = Env_E_field_abs(Env_Ey_plane_xz,Env_Ey_plane_xz_abs);#plt.figure();plt.imshow(Env_Ey_plane_xy_abs.T,aspect="auto",origin="lower")
    fluence_Ey_axis_z   = np.zeros_like(Env_Ey_plane_xz_abs[0,:])
    fluence_Ey_axis_z   = compute_fluence_along_axis(Env_Ey_plane_xz_abs,fluence_Ey_axis_z)
    try:
        two_times_waist_z = compute_width_at_level_times_peak_value(z_axis,fluence_Ey_axis_z,level=1./np.e**2)
        one_ov_e2_width_z.append(two_times_waist_z)
    except:
        one_ov_e2_width_z.append(0.)
    del Ey_on_plane_xz,Env_Ey_plane_xz,Env_Ey_plane_xz_abs,fluence_Ey_axis_z

plt.ion()


one_ov_e2_width_y                       = np.asarray(one_ov_e2_width_y)
one_ov_e2_width_y[one_ov_e2_width_y==0] = np.nan

one_ov_e2_width_z                       = np.asarray(one_ov_e2_width_z)
one_ov_e2_width_z[one_ov_e2_width_z==0] = np.nan

# x-x_focus in the PIC simulation
x_minus_x_focus_PIC                     = (np.asarray(timesteps)
                                           * S.namelist.dt*S.namelist.c_over_omega0
                                           +S.namelist.xmin_minus_LG_focus
                                           -S.namelist.center_laser/S.namelist.meter) 


###  If you don't want to compute again the 1/e^2 widths next time, save then here in .npy ...
np.save('one_ov_e2_width_y.npy',(x_minus_x_focus_PIC, one_ov_e2_width_y) )
np.save('one_ov_e2_width_z.npy',(x_minus_x_focus_PIC, one_ov_e2_width_z) )

###  ... and then read the saved files

#x_minus_x_focus_PIC, one_ov_e2_width_y = np.load('one_ov_e2_width_y.npy')
#x_minus_x_focus_PIC, one_ov_e2_width_z = np.load('one_ov_e2_width_z.npy')


### Plot the data from the Smilei PIC simulation
fig, (ax1, ax2)          = plt.subplots(2, 1, figsize=(4.8,6.93))
ax1.plot(x_minus_x_focus_PIC/1e-6, one_ov_e2_width_y/1e-6,"b",label="Smilei",zorder=0)
ax2.plot(x_minus_x_focus_PIC/1e-6, one_ov_e2_width_z/1e-6,"b",label="Smilei",zorder=0)

#plt.ylabel("1/e^2 amplitude")
#plt.xlabel("x [um]")

###################### Plot the GSA-reconstructed data #########################
from gsa_md.plot_utilities.plot_functions import *
dict_image_preprocessing = np.load('./outputs_cylindrical_GSA_MD/dict_image_preprocessing.npy',allow_pickle=True).item()
dict_mesh                = np.load('./outputs_cylindrical_GSA_MD/dict_mesh.npy'               ,allow_pickle=True).item()
dict_GSA                 = np.load('./outputs_cylindrical_GSA_MD/dict_GSA.npy'                ,allow_pickle=True).item()
fig_1e2_width_y, fig_FWHM_y, fig_1e2_width_z, fig_FWHM_z = plot_diffraction(dict_image_preprocessing,dict_mesh,dict_GSA,plot_cartesian_experimental_fluence=False,plot_cylindrical_experimental_fluence=False,only_markers=True)


# Plot the content of fig_1e2_width_y to ax1
# Copy the content of the first figure into the new subplot (ax1)
for line in fig_1e2_width_y.axes[0].get_lines():
    ax1.scatter(line.get_xdata(), line.get_ydata(), color='red', zorder=line.get_zorder(),label="cylindrical GSA_MD, y")
ax1.set_ylim(0, 70)
ax1.set_xlabel("x-x_focus [um]")
ax1.set_ylabel("1/e^2 width [um]")
ax1.legend()

# Retrieve the data of the diffraction figure
x_minus_x_focus_GSA_MD = line.get_xdata()
one_ov_e2_y_GSA_MD     = line.get_ydata()
 
# Compute absolute differences between every line.get_xdata() and x
diff                   = np.abs(x_minus_x_focus_GSA_MD[:, None] - x_minus_x_focus_PIC[None, :]/1e-6)  # shape: (10, 100)

# Find index of the closest x2 point for each x1
closest_indices        = np.argmin(diff, axis=1)

# Evaluate the abs differece
abs_diff_y             = np.abs((one_ov_e2_width_y[closest_indices]*1e6)-one_ov_e2_y_GSA_MD)*100/(one_ov_e2_y_GSA_MD) # (%)
np.save('relative_abs_error_y.npy',abs_diff_y )

# Plot the content of fig_1e2_width_z to ax2
# Copy the content of the second figure into the new subplot (ax2)
for line in fig_1e2_width_z.axes[0].get_lines():
    ax2.scatter(line.get_xdata(), line.get_ydata(), color='red', zorder=line.get_zorder(),label="cylindrical GSA_MD, z")
ax2.set_ylim(0, 70)
ax2.set_xlabel("x-x_focus [um]")
ax2.set_ylabel("1/e^2 width [um]")
ax2.legend()

# Retrieve the data of the diffraction figure
x_minus_x_focus_GSA_MD = line.get_xdata()
one_ov_e2_z_GSA_MD     = line.get_ydata()
 
# Compute absolute differences between every line.get_xdata() and x
diff                   = np.abs(x_minus_x_focus_GSA_MD[:, None] - x_minus_x_focus_PIC[None, :]/1e-6)  

# Find index of the closest x2 point for each x1
closest_indices        = np.argmin(diff, axis=1)

# Evaluate the abs differece
abs_diff_z             = np.abs((one_ov_e2_width_z[closest_indices]*1e6)-one_ov_e2_z_GSA_MD)*100/(one_ov_e2_z_GSA_MD) # (%)
np.save('relative_abs_error_z.npy',abs_diff_z )

# Adjust layout to ensure no overlapping elements
plt.tight_layout()

# close some figures
[ plt.close(fig)for fig in [fig_1e2_width_y, fig_FWHM_y, fig_1e2_width_z, fig_FWHM_z]]

# Show the plot
plt.show()

########## Compare with the width obtained through analytical propagation ######
 
dict_image_preprocessing = np.load('./outputs_cylindrical_GSA_MD/dict_image_preprocessing.npy',allow_pickle=True).item()
dict_mesh                = np.load('./outputs_cylindrical_GSA_MD/dict_mesh.npy',allow_pickle=True).item()
dict_mode_basis          = np.load('./outputs_cylindrical_GSA_MD/dict_mode_basis.npy',allow_pickle=True).item()
Coeffs_LG_pl             = np.load('./outputs_cylindrical_GSA_MD/Coeffs_MD_iteration_00199.npy')

y_axis_theory            = np.arange(0,4*S.namelist.nr)*S.namelist.length_per_pixel_exp
y_axis_theory            = y_axis_theory-y_axis_theory.max()/2.
z_axis_theory            = y_axis_theory

from gsa_md.mode_basis.laguerre_gauss_modes import *

lambda_0                 = S.namelist.lambda_0
r_mesh_axis_y            = np.abs(y_axis_theory)
theta_mesh_axis_y        = np.array([np.pi if y>0 else 0 for y in y_axis_theory])
r_mesh_axis_z            = np.abs(z_axis_theory)
theta_mesh_axis_z        = np.array([np.pi/2. if z>0 else -np.pi/2. for z in z_axis_theory])

one_ov_e2_width_y_theory = []
one_ov_e2_width_z_theory = []

for i_plane in range(0,np.size(x_minus_x_focus_PIC)):

    Ey_axis_y            = np.zeros_like(r_mesh_axis_y,dtype=complex)
    Ey_axis_z            = np.zeros_like(r_mesh_axis_z,dtype=complex)

    print("Computing analytical 1/e^2 widths at plane ",i_plane," of ",np.size(x_minus_x_focus_PIC))

    dict_mesh_axis_y     = {"plane_x_coordinates": np.array([x_minus_x_focus_PIC[i_plane]]),\
                             "r_mesh"             : r_mesh_axis_y,\
                             "theta_mesh"         : theta_mesh_axis_y} 
    dict_mesh_axis_z     = {"plane_x_coordinates": np.array([x_minus_x_focus_PIC[i_plane]]),\
                             "r_mesh"             : r_mesh_axis_z,\
                             "theta_mesh"         : theta_mesh_axis_z} 

    for l in range(-dict_mode_basis["Max_LG_index_l"], dict_mode_basis["Max_LG_index_l"] + 1):
        # negative l indices are stored with the FFT convention for negative frequencies
        l_index = l if l >= 0 else dict_mode_basis["Max_LG_index_l"] + abs(l)
        # sum the x,r part of the modes with the same l index and different radial p indices
        for p in range(0,dict_mode_basis["Max_LG_index_p"]+1):
            Ey_axis_y   += (Coeffs_LG_pl[p,l_index]\
                         * LG_pl_field_x_r(lambda_0,p,l,dict_mesh_axis_y,dict_mode_basis ,check_total_power_integral=False) \
                         * LG_pl_field_theta(l,dict_mesh_axis_y,dict_mode_basis,check_total_power_integral=False)).flatten()
            Ey_axis_z   += (Coeffs_LG_pl[p,l_index]\
                         * LG_pl_field_x_r(lambda_0,p,l,dict_mesh_axis_z,dict_mode_basis ,check_total_power_integral=False) \
                         * LG_pl_field_theta(l,dict_mesh_axis_z,dict_mode_basis,check_total_power_integral=False)).flatten()

    one_ov_e2_width_y_theory.append(compute_width_at_level_times_peak_value(y_axis,np.square(np.abs(Ey_axis_y)),level=1./np.e**2))
    one_ov_e2_width_z_theory.append(compute_width_at_level_times_peak_value(z_axis,np.square(np.abs(Ey_axis_z)),level=1./np.e**2))

one_ov_e2_width_y_theory = np.array(one_ov_e2_width_y_theory)
one_ov_e2_width_z_theory = np.array(one_ov_e2_width_z_theory)

#plt.figure()
ax1.plot(x_minus_x_focus_PIC[2::]/1e-6,one_ov_e2_width_y_theory[2::]/1e-6,label="analytical propagation",c="r",linestyle = "dashed",dashes=(10,5),zorder=1)
ax2.plot(x_minus_x_focus_PIC[2::]/1e-6,one_ov_e2_width_z_theory[2::]/1e-6,label="analytical propagation",c="r",linestyle = "dashed",dashes=(10,5),zorder=1)

#ax1.legend()
#ax2.legend()

# # Save the combined figure
# #fig.savefig('GSA-MD_vs_Smilei_propagation.pdf',format="pdf",dpi=1000)
