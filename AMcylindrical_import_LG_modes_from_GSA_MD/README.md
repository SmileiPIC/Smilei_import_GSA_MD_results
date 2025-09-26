# Import the GSA-MD results with Smilei

The namelist in this folder runs a PIC simulation that propagates the laser pulse in vacuum,
in cylindrical geometry with azimuthal mode decomposition (https://smileipic.github.io/Smilei/Understand/azimuthal_modes_decomposition.html). The namelist can be easily adapted to run 
the same simulation in 3D.

The repository also includes postprocessing scripts to analyse the simulation results and recreate Figs. ... of ...

### How to use it

First, download and install the `gsa_md` Python library from ...

From ... `git clone` the outputs of the GSA-MD for the benchmark described in ...,
This will create the folder `results_GSA_MD_HZDR_dataset`, which contains the subfolder `outputs_cylindrical_GSA_MD`.
These outputs contain information like the number and parameters of LG modes 
used by the GSA-MD, and the LG mode coefficients.

Create a folder where you will run the PIC simulation. Inside this folder, copy the folder `outputs_cylindrical_GSA_MD`.

In your simulation folder, copy and paste the input namelist of this repository.

Run Smilei using the input namelist. The results shown in Figs. of ... were obtained with v. 5.12 of Smilei,
on CINES Adastra Genoa. On that architecture, using 384 cpu cores, the simulation 
needed ~2.7 hours to complete (configuration not optimized for performances).

To recreate those Figures, use the postprocessing scripts in the repository.

### Contributors

Francesco Massimo created the input namelist, Mohamad Masckala developed the postprocessing scripts. The development of the input namelist benefitted from previous work by Ioaquin Moulanier.

### Acknowledgements
The laser pulse imported by the input namelist uses the results of the GSA-MD obtained using fluence distributions collected during an experiment conducted at the HZDR DRACO facility (https://www.hzdr.de/db/Cms?pOid=40859&pNid=2096&pLang=en), in the frame of the  user  access  programme  in  Laserlab  Europe  (grant agreement no.  871124). We wish to acknowledge Arie Irman, Ulrich Schramm and the HZDR team for fruitful discussions and assistance during the experimental data analysis. 

The development and benchmarking of the input namelist in this repository
benefitted from access to the HPC resources of CINES under the allocation 
2023-A0170510062 (VirtualLaplace) made by GENCI. 
It also benefited from the use of the meso-scale HPC “3Lab Computing” 
hosted at École polytechnique and administrated by the Laboratoire Leprince-Ringuet, 
Laboratoire des Solides Irradiés et Laboratoire pour l’Utilisation des Lasers Intenses.