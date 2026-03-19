###### Author: F. Massimo

###### Plot the energy spectrum of the selected Species at the selected timestep index

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import math
import happi

##### Inputs

species              = "electronfromion"  # tracked species name

timestep_index       = -1  # -1 is for the last timestep, as in a list

nbins_Energy         = 500
Emin_MeV             = 0   # Min energy considered in histogram in MeV
Emax_MeV             = 250 # Max energy considered in histogram in MeV

##### Open the simulation
S                    = happi.Open()

#### Define variables used for unit conversions
try:
    lambda_0         = S.namelist.lambda_0 # laser wavelength, m
except:
    try:
        lambda_0     = S.namelist.lambda0  # laser wavelength, m
    except:
        lambda_0     = 0.8e-6 
c                    = sc.c                              # lightspeed, m/s
omega_0              = 2*math.pi*c/lambda_0              # laser angular frequency, rad/s
eps0                 = sc.epsilon_0                      # Vacuum permittivity, F/m
e                    = sc.e                              # Elementary charge, C
me                   = sc.m_e                            # Electron mass, kg
ncrit                = eps0*omega_0**2*me/e**2           # Plasma critical number density, m-3
c_over_omega0        = lambda_0/2./math.pi               # converts from c/omega0 units to m
electron_mass_MeV    = sc.physical_constants['electron mass energy equivalent in MeV'][0]

from_weight_to_pC    = e * ncrit * (c_over_omega0)**3 / 1e-12

dt                   = S.namelist.dt                     # integration timestep, 1/omega0 units



### Extract the data from the TrackParticles
track_part           = S.TrackParticles(species=species, sort=False)
timesteps            = track_part.getAvailableTimesteps() # timesteps available in this diagnostic
timesteps            = list( dict.fromkeys(timesteps) )   # this is to eliminate doubles in case a checkpoint was used
timesteps.sort()
timestep             = timesteps[-1]
print("Available timesteps in the tracks:", timesteps)

## Extract the spectrum data at each timestep
print("extracting timestep ",timestep,", i.e. c*T = ",timestep*dt*c_over_omega0 /1e-6," um")

track_part           = S.TrackParticles(species=species, sort=False,timesteps=timestep).getData()[timestep]
x                    = track_part["x" ]*c_over_omega0 # x coordinate array, c/omega0 units, each element is one macro-particle
px                   = track_part["px"]               # px momentum array , m_e*c units, each element is one macro-particle
py                   = track_part["py"]               # py momentum array , m_e*c units, each element is one macro-particle
pz                   = track_part["pz"]               # pz momentum array , m_e*c units, each element is one macro-particle
w                    = track_part["w" ]*from_weight_to_pC;print(str(np.sum(w))+" pC") # charge of the macroparticles array, pC

npart                = px.size # number of macro-particles
p                    = np.sqrt((px**2+py**2+pz**2))               # momentum magnitude array, m_e*c units, each element is one macro-particle
E_MeV                = np.sqrt(1+np.square(p))*electron_mass_MeV  # energy array in MeV, each element is one macro-particle

# compute the energy spectrum
hist_energy_axis_MeV = np.linspace(Emin_MeV,Emax_MeV,num=nbins_Energy)   
hist,bin_edges       = np.histogram(E_MeV,bins=hist_energy_axis_MeV,weights=w)
Energy_bin_centres   = [(hist_energy_axis_MeV[i]+hist_energy_axis_MeV[i+1])/2. for i in range(len(hist_energy_axis_MeV)-1)]
bin_width_E          = Energy_bin_centres[1]-Energy_bin_centres[0]
hist                 = hist/bin_width_E

# Plot energy spectrum
linestyle = "-"
linecolor = "r"
linewidth = 2
plt.ion()
plt.figure(1)
plt.plot(Energy_bin_centres,hist,linestyle=linestyle,c=linecolor,linewidth=linewidth)
plt.xlabel("Energy [MeV]")
plt.ylabel("dQ/dE [pC/MeV]")

def weighted_std(values, weights):
    """
    Return the weighted standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return math.sqrt(variance)

avg_energy            = np.average(E_MeV,weights=w)
index_avg_energy      = np.argmin(np.abs(Energy_bin_centres-avg_energy))
rms_energy_spread     = weighted_std(E_MeV,weights=w)


index_0dot1_pC_ov_MeV = np.argmin(np.abs(hist[index_avg_energy::]-0.1))

print("Charge by summing the macro-particle charges                    = ",np.sum(w[~np.isnan(w)]),"pC")
print("Charge by integration of the spectrum                           = ",np.sum(hist)*bin_width_E,"pC")
print("Maximum energy                                                  = ",Energy_bin_centres[index_avg_energy+index_0dot1_pC_ov_MeV]," MeV")
print("Average energy                                                  = ",avg_energy," MeV")
print("Rms energy spread                                               = ",weighted_std(E_MeV,weights=w), "MeV")
print("Relative rms energy spread = Rms energy spread / average energy = ",rms_energy_spread/avg_energy*100," %")


