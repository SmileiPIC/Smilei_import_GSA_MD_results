import happi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmocean


####### Select index of the available outputs to plot
timestep_index = 34

####### Extract the data

# Open the simulation, extract some useful quantities
S              = happi.Open()
lambda_0       = S.namelist.lambda_0 # meter
dx, nx         = S.namelist.dx, S.namelist.nx # dx is in c/omega0 units
dr, nr         = S.namelist.dr, S.namelist.nr # dr is in c/omega0 units

# Extract electron density
Rho_to_plot    = "-Rho_bckgelectron/e-Rho_electronfromion/e"
Rho            = np.asarray(S.Probe.Probe1(Rho_to_plot,timestep_indices=timestep_index,units=["um","1/cm^3"]).getData())[0,:,:]

# Extract longitudinal electric field close to the propagation axis
Ex             = np.asarray(S.Probe.Probe1("Ex",timestep_indices=timestep_index,units=["um","GV/m"]).getData())[0,:,nr+1] 
x_mesh_Ex_um   = (S.Probe.Probe1("Rho_nitrogen5plus/e",units=["um"]).getAxis("axis1"))[:,0]

# Extract the absolute value of the complex envelope of the laser transverse electric field
try: 
    # Simulation with Laser Envelope model: the extracted field is already 
    # the absolute value of the complex envelope of the transverse electric field
    Env_E_abs  = np.asarray(S.Probe.Probe1("Env_E_abs",timestep_indices=timestep_index,units=["um","TV/m"]).getData())[0,:,:]
except: 
    # Simulation without Laser Envelope model:
    # we can only extract the field, 
    # thus the absolute value of its complex envelope must be computed
    # with postprocessing

    E_transverse = np.asarray(S.Probe.Probe1("Ey",timestep_indices=timestep_index,units=["um"]).getData())[0,:,:]
    
    # Function to apply low-pass filter
    fs          = 1.0 / dx  # Sampling frequency based on dx
    cutoff_freq = 0.1       # Cutoff frequency for low-pass filter
    
    from scipy.signal import butter, filtfilt
    def low_pass_filter(signal, cutoff_freq, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freq / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        filtered_signal = filtfilt(b, a, signal)
        return filtered_signal
    
    # Function to find the complex envelope of the transverse electric field
    cos_x = np.cos(np.arange(0,nx)*dx)
    sin_x = np.sin(np.arange(0,nx)*dx)
    
    ## Function to extract the absolute value of the complex envelope 
    ## of the laser transverse electric field
    def from_E_to_Env_E(E_transverse):
        # Input: the transverse electric field E_transverse
        # Output: the complex envelope Env_E of the transverse electric field (with a constant error in the global phase)

        # from the definition of complex envelope, we have
        # E_tranverse = Re[Env_E*exp(i*k0(x-ct))]
        # or in normalized units, neglecting the phase term given by time:
        # E_tranverse = Re[Env_E*exp(i*x)]=Env_E_real(x)*cos(x)-Env_E_imag(x)*sin(x). (2)
        # Multiplying Eq. (2) by cos(x) we have
        # E_tranverse*cos(x) = Env_E_real(x)*cos^2(x)-Env_E_imag(x)*sin(x)*cos(x)=
        #                    # Env_E_real(x)*(1+cos(2x))/2 - Env_E_imag(x)*sin(2x)/2
        # By filtering the higher frequencies centered around 2*x, we can find the Env_E_real(x)/2
        # Multiplying Eq. (2) by sin(x) we have
        # E_tranverse*sin(x) = Env_E_real(x)*cos(x)*sin(x)-Env_E_imag(x)*sin^2(x)=
        #                    = Env_E_real(x)*sin(2*x)/2-Env_E_imag(x)*(1-cos(2x))/2
        # By filtering the higher frequencies entered around 2*x, we can find -Env_E_real(x)/2
        
        Env_E_real = np.zeros_like(E_transverse)
        Env_E_imag = np.zeros_like(E_transverse)
        for i_r in range(0,np.size(E_transverse[0,:])):
            # filter E_transverse * cos_x and multiply by 2 to get the real part of Env_E
            Env_E_real[:,i_r] = 2 * low_pass_filter(E_transverse[:,i_r] * cos_x, cutoff_freq, fs)
            # filter E_transverse * sin_x and multiply by 2 to get the opposite of the imaginary part of Env_E 
            Env_E_imag[:,i_r] = -2 * low_pass_filter(E_transverse[:,i_r] * sin_x, cutoff_freq, fs)  
    
        # Result: the envelope Env_E of the transverse electric field. 
        # The phase at different iterations (given by the neglected time term) is not correct.
        # This error is constant along all the x axis.
        # This does not change the absolute value of |Env_E|, 
        # which is typically what is needed for the diagnostics
        
        return Env_E_real+1j*Env_E_imag 
        
    Env_E        = from_E_to_Env_E(E_transverse)
    Env_E_abs    = np.abs(Env_E)

####### Plot electron density

# Extract limits of the window
xmax_um   = (S.Probe.Probe1("Ex",units=["um","GV/m"]).getAxis("axis1"))[-1,0]
ymin_um   = (S.Probe.Probe1("Ex",units=["um","GV/m"]).getAxis("axis1"))[-1,1]
ymax_um   = -ymin_um    
extent    = [0,xmax_um,ymin_um,ymax_um] 

# Create new figure
fig, ax1  = plt.subplots(figsize=(10,6))

# Create custom colormap for electron density
from matplotlib.colors import LinearSegmentedColormap
colors    = [(0.,0.,0.0), (0.00,0.5,1),(0,1,1),(1,1,1)]
n_bins    = 512  # Discretizes the interpolation into bins
cmap_name = 'ne_cmap'
ne_cmap   = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

# Plot the electron density
im=ax1.imshow(Rho.T/1e18,extent=extent,aspect="auto",origin="lower",cmap=ne_cmap,vmin=0,vmax=5.5)
ax1.set_xlabel("x-ct [um]")
ax1.set_xlim(0,70)
ax1.set_ylabel("y [um]")
ax1.set_ylim(-70,70)
clb = plt.colorbar(im, ax=ax1, orientation='vertical', fraction=0.045, pad=0.05, aspect=30)
clb.set_label('n_e [10^18 cm^-3]', fontsize=10)

####### Overlay the absolute value of the laser field envelope with variable transparency

## Function to have a pixel-wise transparency based on the field value,
## derived from Rémi Lehe's implementation
def Imshow(data, tvmin=None, tvmax=None, tmax=None, colorbar_title=None, **kwargs):
    if 'vmax' in kwargs:
        vmax = kwargs['vmax']
    else:
        vmax = data.max()
    if 'vmin' in kwargs:
        vmin = kwargs['vmin']
    else:
        vmin = data.min()
    if tvmax is None:
        tvmax = vmax
    if tvmin is None:
        tvmin = vmin    
    if tmax is None:
        tmax = 1.

    cmap = kwargs['cmap']

    # Rescale the data to get the transparency and color
    color = (data - vmin) / (vmax - vmin)
    color = np.clip(color, 0., 1.)
    transparency = tmax * ((data - tvmin) / (tvmax - tvmin))
    transparency = np.clip(transparency, 0., 1.)
    transparency = tmax * transparency

    rgba_data = cmap(color)
    rgba_data[:,:,3] = transparency

    # Draw on current axes
    im = plt.imshow(rgba_data, **kwargs)

    if colorbar_title:
        cbar = plt.colorbar(im, orientation='vertical', fraction=0.045, pad=0.15, aspect=30)
        cbar.set_label(colorbar_title, fontsize=10)
    return im

# Plot the laser, with transparency
im2 = Imshow(
    Env_E_abs.T, origin="lower", aspect="auto", extent=extent, 
    vmin=0.0, vmax=3., cmap=mpl.colormaps["Reds_r"], tvmin=0.0, tvmax=1.5,
    colorbar_title="|E_y| [m_e*omega_0*c/e]", zorder=2)

####### Plot Ex close to the propagation axis, in GV/m 

ax2 = ax1.twinx()

# Make the background transparent
ax2.set_facecolor('none')  
ax2.patch.set_alpha(0)

# Plot the 1D Ex field
line_color = "limegreen"
ax2.plot(x_mesh_Ex_um, Ex, color=line_color, linewidth=2, label="Ex", zorder=10)

ax2.set_ylabel("E_x [GV/m]", color=line_color,fontsize=16)
ax2.tick_params(axis='y', colors=line_color)
ax2.spines['right'].set_color(line_color)

# Set the x axis limit
ax2.set_xlim(0,70)
# Set the y axis limit for the Ex field
ax2.set_ylim(-400,400)


 