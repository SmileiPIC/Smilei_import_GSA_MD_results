###### Author: F. Massimo, I. Moulanier, T. L. Steyn

###### An input namelist for the PIC code Smilei https://smileipic.github.io/Smilei/
###### that imports a laser pulse (amplitude and phase of its fields)
###### reconstructed from fluence measurements through the GSA-MD algorithm, 
###### using a Laguerre-Gauss (LG) mode decomposition in cylindrical geometry
###### with azimuthal mode decomposition.

import numpy as np
import scipy.constants as sc

## Load the results of the GSA-MD laser field reconstruction
dict_mode_basis          = np.load('outputs_cylindrical_GSA_MD/dict_mode_basis.npy',allow_pickle=True).item()
dict_GSA                 = np.load('outputs_cylindrical_GSA_MD/dict_GSA.npy',allow_pickle=True).item()
Coeffs_LG_pl             = np.load('outputs_cylindrical_GSA_MD/Coeffs_MD_iteration_00199.npy')

# True  : use an idealized Gaussian beam fit, with hard-coded parameters
# False : use the laser field reconstructed by the GSA-MD
use_Gaussian_beam_fit    = False #True 

#########################  Physical constants #########################

e                        = sc.e                               # Elementary charge, C
me                       = sc.m_e                             # Electron mass, kg
eps0                     = sc.epsilon_0                       # Vacuum permittivity, F/m
c                        = sc.c                               # lightspeed, m/s

lambda_0                 = dict_GSA["lambda_0"]               # laser wavelength, m
omega_0                  = 2*np.pi*c/lambda_0                 # laser angular frequency, rad/s
ncrit                    = eps0*omega_0**2*me/e**2            # Plasma critical number density, m-3
c_over_omega0            = lambda_0/2./np.pi                  # converts from c/omega0 units to m
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
dr                       = 0.35*um                            # transverse mesh resolution, normalized
dt                       = 0.96*dx/c_normalized               # integration timestep, normalized

##### Simulation window size
nx                       = 4608                               # number of mesh points in the longitudinal direction
nr                       = 512                                # number of mesh points in the radial direction (half plane)
# number of mesh points in the transverse direction
Lx                       = nx * dx                            # longitudinal size of the simulation window
Lr                       = nr * dr                            # transverse size of the simulation window (half plane)

##### Patches parameters (parallelization)
npatch_x                 = 512
npatch_r                 = 4

##### Number of azimuthal modes
number_of_AM             = 2 if use_Gaussian_beam_fit else 5

##### Total simulation time
T_sim                    = 6101*um

########################## Main simulation definition block #########################

Main(
    geometry                       = "AMcylindrical",
    number_of_AM                   = number_of_AM,
    interpolation_order            = 2,

    timestep                       = dt,
    simulation_time                = T_sim,

    cell_length                    = [dx, dr],
    grid_length                    = [ Lx,  Lr],
    number_of_patches              = [npatch_x,npatch_r], 

    EM_boundary_conditions         = [["silver-muller"],["PML"],],
    number_of_pml_cells            = [[0,0],[20,20]],

    solve_poisson                  = False,
    solve_relativistic_poisson     = False,
    print_every                    = 100,
    reference_angular_frequency_SI = omega_0,     # Necessary for the physical. 
    random_seed                    = smilei_mpi_rank,
    use_BTIS3_interpolation        = True,
    maxwell_solver                 = "Terzani"
)

######################### Checkpoints #########################

Checkpoints(
    dump_step            = int(1000*um/dt),
    exit_after_dump      = False,
    keep_n_dumps         = 2,
)

######################### Load balancing (for parallelization) #########################
LoadBalancing(
    initial_balance      = False,
    every                = 100,
    cell_load            = 1.,
    frozen_particle_load = 0.1
)

############################# Laser pulse definition #############################


#### Longitudinal profile

# FWHM duration in field, i.e. sqrt(2) time the FWHM duration in intensity
laser_fwhm_s = 32.*np.sqrt(2)*1e-15   # seconds
laser_fwhm   = laser_fwhm_s/1e-15*fs  # 1/omega0 units
# The laser peak will enter the left border of the window after this normalized time
center_laser = 1.8*laser_fwhm

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
x_focus_LG_modes          = 3500.*um

##### Define the laser from its LG mode decomposition

# It is assumed that the laser pulse is linearly polarized along y and propagates along the direction +x

# normalized laser frequency
omega                     = (2*np.pi*c/lambda_0)/omega_0

# The LG modes of GSA-MD are normalized with the lengths in meters, so they scale as 1/meter,
# while the laser creator uses the LG modes normalized with the lengths in c/omega0 units.
# Therefore, the coefficients in Smilei units are the rescaled coefficients
# normalized by E0 = m_e*omega0*c/e and divided by c_over_omega0.
Coeffs_LG_pl_Smilei_units = Coeffs_LG_pl_rescaled/E0/c_over_omega0

if (use_Gaussian_beam_fit ==True): # use an idealized Gaussian beam fit, with hard-coded parameters
    w0 = 18*um
    a0 = 2.035 
    LaserGaussianAM(
        box_side              = "xmin",
        a0                    = a0,
        omega                 = omega,
        focus                 = [x_focus_LG_modes],
        waist                 = w0,
        ellipticity           = 0., polarization_phi = 0., # linear polarization along y
        time_envelope         = tgaussian(center=center_laser,fwhm=laser_fwhm)
    )
else: # use the laser field reconstructed by the GSA-MD
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

####################### Plasma profile definition #############################

### Plasma radius in the simulation window
R_plasma                     = 0.98*Lr # normalized units

### Read the spline coefficients for the electrons
Data_electron_density_spline = np.genfromtxt("electron_density_splines.csv", delimiter=";")
x_splines                    = Data_electron_density_spline[:,0]
coeffs_spline_electron       = [Data_electron_density_spline[:,1]]
coeffs_spline_electron.append(Data_electron_density_spline[:,2])
coeffs_spline_electron.append(Data_electron_density_spline[:,3])
coeffs_spline_electron.append(Data_electron_density_spline[:,4])

### Read the spline coefficients for the nitrogen
Data_nitrogen_density_spline = np.genfromtxt("nitrogen_density_splines.csv", delimiter=";")
x_splines                    = Data_nitrogen_density_spline[:,0]
coeffs_spline_nitrogen       = [Data_nitrogen_density_spline[:,1]]
coeffs_spline_nitrogen.append(Data_nitrogen_density_spline[:,2])
coeffs_spline_nitrogen.append(Data_nitrogen_density_spline[:,3])
coeffs_spline_nitrogen.append(Data_nitrogen_density_spline[:,4])

### function to reconstruct the density profile from splines
def reconstruct_density_from_splines(x,x_splines,coeffs_spline):
    x = np.asarray(x)
    # Find interval index for each spline
    i = np.searchsorted(x_splines, x) - 1
    i = np.clip(i, 0, np.size(x_splines)-2)
    # Local coordinate in the spline
    dx = x - x_splines[i]
    # Evaluate polynomial
    return (coeffs_spline[0])[i] + (coeffs_spline[1])[i]*dx + (coeffs_spline[2])[i]*(dx**2) + (coeffs_spline[3])[i]*(dx**3)

def dens_nitrogen(x):
	# x is in meters, the returned density is in critical density units
    return reconstruct_density_from_splines(x,x_splines,coeffs_spline_nitrogen)

def dens_Funct_Nitrogen(x,r): ### x received in code units, calculations done in mm in sub functions
	# x, r are normalized
    radial_profile = np.ones_like(r)
    radial_profile = np.where((r / meter) > R_plasma/meter, 0.0, radial_profile)
    return radial_profile*dens_nitrogen(x/meter)

def dens_electrons(x):
	# x is in m, the returned density is in critical density units
    return reconstruct_density_from_splines(x,x_splines,coeffs_spline_electron)

def dens_Funct_Electrons(x,r): ### x received in code units, calculations done in mm in sub functions
	# x, r are normalized
    radial_profile = np.ones_like(r)
    radial_profile = np.where((r / meter) > R_plasma/meter, 0.0, radial_profile)
    return radial_profile*dens_electrons(x/meter)

######################### Species definition #########################

## Background Electron Species (H and first 5 levels of N)
Species(
    name 	                = "bckgelectron",
    position_initialization = "regular",
    momentum_initialization = "cold",
    particles_per_cell      = 32, # [along x,along r,along theta], the product of these three numbers must be equal to particles_per_cell, N_theta>= 4*number_of_AM 
    regular_number          = [1,2,16], 
    c_part_max              = 1.0,
    mass                    = 1.0,
    charge                  = -1.0,
    number_density          = dens_Funct_Electrons,
    mean_velocity           = [0.0, 0.0, 0.0],
    temperature             = [0.,0.,0.],
    time_frozen             = 0.0,
    pusher                  = "borisBTIS3",
    boundary_conditions     = [["remove", "remove"],["remove", "remove"],],
)

# Nitrogen N5+ ions (i.e. already ionized up to the first two ionization levels over seven)
Species(
    name                    = "nitrogen5plus",
    position_initialization = "regular", 
    momentum_initialization = "cold",
    particles_per_cell      = 32, 
    regular_number          = [1,2,16], 
    atomic_number           = 7, # Nitrogen
    ionization_model        = "tunnel",
    ionization_electrons    = "electronfromion",
    maximum_charge_state    = 7,
    c_part_max              = 1.0,
    mass                    = 7.*mp_over_me + 7.*mn_over_me + 2.*me_over_me,
    charge                  = 5.0,
    number_density          = dens_Funct_Nitrogen,
    mean_velocity           = [0., 0., 0.],
    time_frozen             = 2*T_sim,                           # for static species i.e., immobile ions. Here, N5+ ions will remain static upto time = 2*T_sim
    pusher                  = "borisBTIS3",
    boundary_conditions     = [ ["remove", "remove"], ["reflective", "remove"],],
)


#### define the electron bunch, i.e., generated from N5+ ions
Species( 
    name                    = "electronfromion",
    position_initialization = "regular",
    momentum_initialization = "cold",
    particles_per_cell      = 0,
    c_part_max              = 1.0,
    mass                    = 1.0,
    charge                  = -1.0,
    number_density          = 0.,  
    pusher                  = "borisBTIS3", 
    boundary_conditions     = [["remove", "remove"],["remove", "remove"], ],
)  


############################# Diagnostics #############################

field_lists_forprobes       = ['Ex','Ey','Bz',"BzBTIS3",'Rho','Rho_bckgelectron','Rho_nitrogen5plus','Rho_electronfromion']

# 1D probe parallel and close to the propagation axis x
DiagProbe(
    every   = int(100*um/dt), #diag_animation_timesteps,
    origin  = [0., 1*dr, 0.],
    corners = [[Main.grid_length[0], 1*dr, 0.]],
    number  = [nx],
    fields  = field_lists_forprobes,
)

# 2D probe on the xy plane
DiagProbe(
    every   = int(100*um/dt), #diag_animation_timesteps, #1000,
    origin  = [0., -Main.grid_length[1], 0.],
    corners = [
              [Main.grid_length[0], -Main.grid_length[1], 0.],
              [0., Main.grid_length[1], 0.],
              ],
    number  = [nx,2*nr],
    fields  = field_lists_forprobes,
)

# Field diagnostic to reconstruct the field at an arbitrary angle
DiagFields(
    every   = int(200*um/dt),
    fields  = ["El", "Er", "Et", 'Rho_bckgelectron', 'Rho_electronfromion'],
)

######################### Track particles diagnostic

# Diag to track the particles
def my_filter(particles):
    return ((particles.px>20.))

DiagTrackParticles(
    species    = "electronfromion", # species to be tracked
    every      = int(100*um/dt),
    filter     = my_filter,
    attributes = ["x","y", "z", "px", "py", "pz","weight"]
)

DiagTrackParticles(
    species    = "bckgelectron", # species to be tracked
    every      = int(100*um/dt),
    filter     = my_filter,
    attributes = ["x","y", "z", "px", "py", "pz","weight"]
)
