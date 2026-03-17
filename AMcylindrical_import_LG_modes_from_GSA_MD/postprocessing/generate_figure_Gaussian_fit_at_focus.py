###### Author: F. Massimo, M. Masckala

###### Plot lineouts of the reconstructed square of the laser field and of a Gaussian fit at the focal plane


from gsa_md.mode_basis.laguerre_gauss_modes  import *

import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt

path                     = '/Users/francescomassimo/Codes/GSA-LG-MD/Datasets_and_scripts_GSA_MD_publication/'
## Load the results of the GSA-MD laser field reconstruction
dict_mode_basis          = np.load(path+'outputs_cylindrical_GSA_MD/dict_mode_basis.npy',allow_pickle=True).item()
dict_GSA                 = np.load(path+'outputs_cylindrical_GSA_MD/dict_GSA.npy',allow_pickle=True).item()
Coeffs_LG_pl             = np.load(path+'outputs_cylindrical_GSA_MD/Coeffs_MD_iteration_00199.npy')

#########################  Physical constants #########################

e                        = sc.e                               # Elementary charge, C
me                       = sc.m_e                             # Electron mass, kg
eps0                     = sc.epsilon_0                       # Vacuum permittivity, F/m
c                        = sc.c                               # lightspeed, m/s
lambda_0                 = dict_GSA["lambda_0"]               # laser wavelength, m
omega_0                  = 2*np.pi*c/lambda_0                 # laser angular frequency, rad/s
ncrit                    = eps0*omega_0**2*me/e**2            # Plasma critical number density, m-3
E0                       = me*omega_0*c/e                     # reference electric field, V/m

##### Variables used for unit conversions
um                       = 1.e-6/c_over_omega0                # 1 micron in normalized units
fs                       = 1.e-15*omega_0                     # 1 femtosecond in normalized units
meter                    = 1./c_over_omega0                   # 1 meter

#########################  Simulation parameters #########################

##### Simulation window size and resolution along r
nx                       = 5120                               # number of mesh points in the longitudinal direction
nr                       = 256                                # number of mesh points in the radial direction (half plane)

# FWHM duration in field, i.e. sqrt(2) time the FWHM duration in intensity
laser_fwhm_s             = 32.*np.sqrt(2)*1e-15

# Compute the total energy of the LG mode contributions, using the GSA-MD coefficients given by the GSA-MD.
# These coefficients are in the units of the fluence image used for the reconstruction.
# The total energy can be computed with the Parseval theorem, because the integral of the intensity of each LG mode 
# at each x plane is normalized to be 1.
total_energy_LG_reconstruction = 0.

# use Parseval theorem to compute the total energy of the reconstruction
for p in range(0, dict_mode_basis["Max_LG_index_p"] + 1):
    for l in range(-dict_mode_basis["Max_LG_index_l"], dict_mode_basis["Max_LG_index_l"] + 1):
        if l>=0:
            l_index = l 
        else:
            l_index = dict_mode_basis["Max_LG_index_l"]+np.abs(l)
        total_energy_LG_reconstruction += np.abs( Coeffs_LG_pl[p, l_index] )**2

laser_energy_focal_plane = 2.2 # J

# Integral of the |time envelope|^2 (which is normalized to 1) in dt;
# for a gaussian this is equal to (sqrt(2pi)*sigma_intensity)^2
integral_time_seconds    = np.sqrt(2*np.pi)*(laser_fwhm_s/np.sqrt(2))/(2*np.sqrt(2*np.log(2)))

# scaling factor to have the desired energy
K_square                 = laser_energy_focal_plane/0.5/eps0/total_energy_LG_reconstruction/(c*integral_time_seconds)
Coeffs_LG_pl_rescaled    = Coeffs_LG_pl * np.sqrt(K_square)

r_mesh                   = np.arange(nr)*dr+dr/2. # Assumes dual and 2 ghost cells per direction
r_mesh                   = r_mesh/meter                       # convert to meter to use the LG function library

dict_mesh_plane_xmin     = {"plane_x_coordinates":np.array([0.]),"r_mesh":r_mesh} 

## Compute the reconstruct the field on the y and z axis

total_reconstruction_plus_y  = np.zeros_like(r_mesh,dtype=complex)
total_reconstruction_minus_y = np.zeros_like(r_mesh,dtype=complex)
total_reconstruction_plus_z  = np.zeros_like(r_mesh,dtype=complex)
total_reconstruction_minus_z = np.zeros_like(r_mesh,dtype=complex)

for l in range(-dict_mode_basis["Max_LG_index_l"], dict_mode_basis["Max_LG_index_l"] + 1):
    l_index = l if l >= 0 else dict_mode_basis["Max_LG_index_l"] + abs(l)
    for p in range(0,dict_mode_basis["Max_LG_index_p"]+1):
        theta = 0.
        total_reconstruction_plus_y  += np.exp(1j*l*theta)*(Coeffs_LG_pl_rescaled [p,l_index]/E0)*LG_pl_field_x_r(lambda_0,p,l,dict_mesh_plane_xmin,dict_mode_basis,check_total_power_integral=False)[0,:]
        theta = np.pi
        total_reconstruction_minus_y += np.exp(1j*l*theta)*(Coeffs_LG_pl_rescaled [p,l_index]/E0)*LG_pl_field_x_r(lambda_0,p,l,dict_mesh_plane_xmin,dict_mode_basis,check_total_power_integral=False)[0,:]
        theta = np.pi/2.
        total_reconstruction_plus_z  += np.exp(1j*l*theta)*(Coeffs_LG_pl_rescaled [p,l_index]/E0)*LG_pl_field_x_r(lambda_0,p,l,dict_mesh_plane_xmin,dict_mode_basis,check_total_power_integral=False)[0,:]
        theta = 3.*np.pi/2.
        total_reconstruction_minus_z += np.exp(1j*l*theta)*(Coeffs_LG_pl_rescaled [p,l_index]/E0)*LG_pl_field_x_r(lambda_0,p,l,dict_mesh_plane_xmin,dict_mode_basis,check_total_power_integral=False)[0,:]

total_reconstruction_y   = np.hstack([np.flip(total_reconstruction_minus_y),total_reconstruction_plus_y])
total_reconstruction_z   = np.hstack([np.flip(total_reconstruction_minus_z),total_reconstruction_plus_z])

# Store the field of a Gaussian fit
dict_mode_basis_Gaussian = {"Max_LG_index_l":0,"Max_LG_index_p":0,"waist_0":17.6e-6,"x_focus":0.,"LG_mode_type":"sinusoidal"}
a0                       = np.amax(np.abs(total_reconstruction_y))
gaussian_fit  = np.zeros_like(r_mesh,dtype=complex)
gaussian_fit  = LG_pl_field_x_r(lambda_0,0,0,dict_mesh_plane_xmin,dict_mode_basis_Gaussian,check_total_power_integral=False)[0,:]
gaussian_fit  = a0*gaussian_fit/np.amax(np.abs(gaussian_fit))

# plot
gaussian_fit_y = np.hstack([np.flip(gaussian_fit),gaussian_fit])
gaussian_fit_z = gaussian_fit_y

# Plot lineout along y axis
plt.ion()
plt.subplot(121)
plt.plot(np.hstack([np.flip(-r_mesh),r_mesh])/1e-6,np.square(np.abs(total_reconstruction_y)),label="full reconstruction",color="r")
plt.plot(np.hstack([np.flip(-r_mesh),r_mesh])/1e-6,np.square(np.abs(gaussian_fit_y)),label="Gaussian fit",linestyle="--",color="k")
plt.xlim(-100,100)
plt.xlabel("y [um]")
plt.ylabel("[E_y/(m_e*omega0*c/e)]^2")
#plt.legend()

# Plot lineout along z axis
plt.subplot(122)
plt.plot(np.hstack([np.flip(-r_mesh),r_mesh])/1e-6,np.square(np.abs(total_reconstruction_z)),label="full reconstruction",color="r")
plt.plot(np.hstack([np.flip(-r_mesh),r_mesh])/1e-6,np.square(np.abs(gaussian_fit_z)),label="Gaussian fit",linestyle="--",color="k")
plt.xlabel("z [um]")
#plt.ylabel("[E_y/(m_e*omega0*c/e)]^2")
plt.xlim(-100,100)
plt.legend()






