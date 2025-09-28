# Import the GSA-MD results with Smilei

This repository will contain input namelists for the open source Particle in Cell 
(PIC) code Smilei https://smileipic.github.io/Smilei/index.html, that import a 
laser pulse field reconstructed with the Gerchberg-Saxton Algorithnm with Mode Decomposition (GSA-MD)

To do this, the namelists import the Laguerre-Gauss (LG) [or Hermite-Gauss (HG)] mode coefficients found by the GSA-MD 
and reconstructs the field following the definition of LG decomposition,
using the same definition of the LG modes used by the GSA-MD.

The namelists runs a PIC simulation that propagates the laser pulse in vacuum.

The repository also includes postprocessing scripts to analyse the simulation results.

### How to use it
First, download and install the `gsa_md` Python library from [https://github.com/GSA-MD/GSA-MD](https://github.com/GSA-MD/GSA-MD).

Then, follow the instructions in the README files of the chosen use-case.

### Contributors
Francesco Massimo created the input namelist for the case in cylindrical geometry with azimuthal mode decomposition, Mohamad Masckala developed the related postprocessing scripts. 
The development of the input namelist benefitted from previous work by Ioaquin Moulanier.

### Acknowledgements
The development and benchmarking of the input namelist in cylindrical geometry in this repository
benefitted from access to the HPC resources of CINES under the allocation 
2023-A0170510062 (VirtualLaplace) made by GENCI. 
It also benefited from the use of the meso-scale HPC “3Lab Computing” 
hosted at École polytechnique and administrated by the Laboratoire Leprince-Ringuet, 
Laboratoire des Solides Irradiés et Laboratoire pour l’Utilisation des Lasers Intenses.
