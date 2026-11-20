###### Author: F. Massimo

###### An input namelist for the PIC code Smilei https://smileipic.github.io/Smilei/
###### that imports a laser pulse (amplitude and phase of its fields)
###### reconstructed from fluence measurements through the GSA-MD algorithm, 
###### using a Laguerre-Gauss (LG) mode decomposition in cylindrical geometry
###### with azimuthal mode decomposition.

import numpy as np
import scipy.constants as sc

## Load the results of the GSA-MD laser field reconstruction
dict_image_preprocessing = np.load('outputs_cylindrical_GSA_MD/dict_image_preprocessing.npy',allow_pickle=True).item()
dict_mode_basis          = np.load('outputs_cylindrical_GSA_MD/dict_mode_basis.npy',allow_pickle=True).item()
dict_GSA                 = np.load('outputs_cylindrical_GSA_MD/dict_GSA.npy',allow_pickle=True).item()
Coeffs_LG_pl             = np.load('outputs_cylindrical_GSA_MD/Coeffs_MD_iteration_00199.npy')
length_per_pixel_exp     = dict_image_preprocessing["length_per_pixel"] # meters

#########################  Physical constants #########################

e                        = sc.e                               # Elementary charge, C
me                       = sc.m_e                             # Electron mass, kg
eps0                     = sc.epsilon_0                       # Vacuum permittivity, F/m
c                        = sc.c                               # lightspeed, m/s

lambda_0                 = dict_GSA["lambda_0"]               # laser wavelength, m
omega_0                  = 2*np.pi*c/lambda_0               # laser angular frequency, rad/s
ncrit                    = eps0*omega_0**2*me/e**2            # Plasma critical number density, m-3
c_over_omega0            = lambda_0/2./np.pi                # converts from c/omega0 units to m
reference_frequency      = omega_0                            # reference frequency, s-1
E0                       = me*omega_0*c/e                     # reference electric field, V/m

##### Variables used for unit conversions
c_normalized             = 1.                                 # speed of light in vacuum in normalized units
um                       = 1.e-6/c_over_omega0                # 1 micron in normalized units
mm                       = 1.e-3/c_over_omega0                # 1 mm in normalized units
me_over_me               = 1.0                                # normalized electron mass
mp_over_me               = sc.proton_mass / sc.electron_mass  # normalized proton mass
mn_over_me               = sc.neutron_mass / sc.electron_mass # normalized neutron mass
fs                       = 1.e-15*omega_0                     # 1 femtosecond in normalized units
mm_mrad                  = um                                 # 1 millimeter-milliradians in normalized units
pC                       = 1.e-12/e                           # 1 picoCoulomb in normalized units
meter                    = 1./c_over_omega0                   # 1 meter
second                   = 1./omega_0                         # 1 second
#
#########################  Simulation parameters #########################

##### Mesh resolution
dx                       = 0.015915*um                        # longitudinal mesh resolution, normalized
dr                       = 0.7*um                             # transverse mesh resolution, normalized
dt                       = 0.98*dx/c_normalized               # integration timestep, normalized

##### Simulation window size
nx                       = 5120                               # number of mesh points in the longitudinal direction
nr                       = 256                                # number of mesh points in the radial direction (half plane)
# number of mesh points in the transverse direction
Lx                       = nx * dx                            # longitudinal size of the simulation window
Lr                       = nr * dr                            # transverse size of the simulation window (half plane)

##### Patches parameters (parallelization)
npatch_x                 = 512
npatch_r                 = 32

##### Number of azimuthal modes
number_of_AM             = 5

##### Total simulation time
T_sim                    = 4201*um

########################## Main simulation definition block #########################

Main(
    geometry                       = "AMcylindrical",
    number_of_AM                   = number_of_AM, #10,
    interpolation_order            = 2,

    timestep                       = dt,
    simulation_time                = T_sim,

    cell_length                    = [dx, dr],
    grid_length                    = [ Lx,  Lr],
    number_of_patches              = [npatch_x,npatch_r], 

    EM_boundary_conditions         = [["silver-muller"],["PML"],],
    number_of_pml_cells            = [[0,0],[20,20]],

    solve_poisson                  = False,
    #solve_relativistic_poisson = True,
    print_every                    = 100,
    reference_angular_frequency_SI = omega_0,     # Necessary for the physical. 
    random_seed                    = smilei_mpi_rank,
    #use_BTIS3_interpolation        = True,
    maxwell_solver                 = "Terzani"
)


######################### Load balancing (for parallelization) #########################
#LoadBalancing(
#     initial_balance = False,
#         every = 500,
#     cell_load = 1.,
#     frozen_particle_load = 0.1
#)

############################# Laser pulse definition #############################


#### Longitudinal profile

# FWHM duration in field, i.e. sqrt(2) time the FWHM duration in intensity
laser_fwhm_s = 32.*np.sqrt(2)*1e-15
laser_fwhm   = laser_fwhm_s/1e-15*fs
# The laser peak will enter the left border of the window after this normalized time
center_laser = Lx/2.

# Compute the total energy of the LG mode contributions, using the GSA-MD coefficients given by the GSA-MD.
# These coefficients are in the units of the fluence image used for the reconstruction.
# The total energy can be computed with the Parseval theorem, because the integral of the intensity of each LG mode
# at each x plane is normalized to be 1.
total_energy_LG_reconstruction = np.sum( np.abs(Coeffs_LG_pl)**2 )

laser_energy_focal_plane = 2.2 # J

# Integral of the |time envelope|^2 (which is normalized to 1) in dt;
# for a gaussian this is equal to (sqrt(2pi)*sigma_intensity)^2
integral_time_seconds    = np.sqrt(2*np.pi)*(laser_fwhm_s/np.sqrt(2))/(2*np.sqrt(2*np.log(2)))

# scaling factor to have the desired energy
K_square                 = laser_energy_focal_plane/0.5/eps0/total_energy_LG_reconstruction/(c*integral_time_seconds)

Coeffs_LG_pl_rescaled    = Coeffs_LG_pl*np.sqrt(K_square)

#######

# The laser pulse will be injected from the border x = 0.

# Focal plane of the LG modes, in simulation coordinates (normalized units)
x_focus_LG_modes          = 2100.*um

##### Define the laser from its LG mode decomposition

# It is assumed that the laser pulse is linearly polarized along y and propagates along the direction +x

# normalized laser frequency
omega                     = (2*np.pi*c/lambda_0)/omega_0

# The LG modes of GSA-MD are normalized with the lengths in meters, so they scale as 1/meter,
# while the laser creator uses the LG modes normalized with the lengths in c/omega0 units.
# Therefore, the coefficients in Smilei units are the rescaled coefficients
# normalized by E0 = m_e*omega0*c/e and divided by c_over_omega0.
Coeffs_LG_pl_Smilei_units = Coeffs_LG_pl_rescaled/E0/c_over_omega0

# The carrier at the left border x = 0 is exp(-1j*omega*(t-center_laser)),
# i.e. exp(-1j*omega*t) with phase_offset = -omega*center_laser
LaserFromLaguerreGaussModeDecompositionAM(
    box_side              = "xmin",
    omega                 = omega,
    focus                 = [x_focus_LG_modes],
    waist                 = dict_mode_basis["waist_0"]*meter,
    coefficients          = Coeffs_LG_pl_Smilei_units,
    polarization_phi      = 0.,
    ellipticity           = 0.,  # linear polarization along y
    time_envelope         = tgaussian(center=center_laser,fwhm=laser_fwhm),
    phase_offset          = -omega*center_laser
)

##### Moving window
MovingWindow(
    time_start            = Lx,          # time_start_moving_window,
    velocity_x            = c_normalized # propagation speed of the moving window along the positive x direction, in c units
)

############################# Diagnostics #############################

field_lists_forprobes     = ["Ey"]


# 1D probe near the propagation axis
DiagProbe(
    every                 = int(25*um/dt),
    origin                = [0., 1*dr, 0.],
    corners               = [[Lx, 1*dr, 1.*dr]],
    number                = [nx],
    fields                = field_lists_forprobes,
)

# 2D probe on the xy plane

side_2D_probe             = 100e-6 # m

DiagProbe(
    every                 = int(25*um/dt),
    origin                = [ 0., -side_2D_probe*meter, 0.],
    corners               = [[Lx, -side_2D_probe*meter, 0.],
                             [0.,  side_2D_probe*meter, 0.],],
    number                = [nx,int(2*side_2D_probe/length_per_pixel_exp)],
    fields                = field_lists_forprobes,
)

# 2D probe on the xz plane
DiagProbe(
    every                 = int(25*um/dt), 
    origin                = [ 0., 0., -side_2D_probe*meter],
    corners               = [[Lx, 0., -side_2D_probe*meter],
                             [0., 0., side_2D_probe*meter ],],
    number                = [nx,int(2*side_2D_probe/length_per_pixel_exp)],
    fields                = field_lists_forprobes,
)

######################### 2D probe on the yz plane
# This probe will be activated when the laser is close to its focal plane
# this probe is used to check that the laser reconstructed by the GSA-MD 
# has correctly been propagated until the focus of the LG modes


# The laser peak at t=0 is at x=-center_laser.
# When the moving window (moving at speed c) starts moving at t=MovingWindow.time_start,
# this will be the approximated position of the laser peak.
# Therefore, the transverse 2D probe x position will be defined here:
laser_x_in_mov_window     = -center_laser + c_normalized*MovingWindow.time_start
# Therefore, the laser peak (moving at ~c in vacuum) will need approximately 
# this time to arrive to that position:
time_laser_at_LG_x_focus  = MovingWindow.time_start + (x_focus_LG_modes-laser_x_in_mov_window)/c_normalized

DiagProbe(
    every                 = [int(time_laser_at_LG_x_focus/dt-5), int(time_laser_at_LG_x_focus/dt+5), 1],
    origin                = [  laser_x_in_mov_window,-side_2D_probe*meter, -side_2D_probe*meter],
    corners               = [ [laser_x_in_mov_window, side_2D_probe*meter, -side_2D_probe*meter], 
                              [laser_x_in_mov_window,-side_2D_probe*meter,  side_2D_probe*meter] ],
    number                = [int(2*side_2D_probe/length_per_pixel_exp),int(2*side_2D_probe/length_per_pixel_exp)],
    fields                = field_lists_forprobes,
)
