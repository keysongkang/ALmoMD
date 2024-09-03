# User Manual

Here, parameters employed in __input.in__ are described.

## Input Parameters

### [Active learning types]
1) __calc_type__ (str): *active* (default), *random*

- Type of the data-sampling. *active* initiates an active learning scheme. *random* triggers a random sampling.

2) __al_type__ (str): *energy*, *force*, *force_max* (default)

- Source of the uncertainty. Ensemble uncertainty is obtained from potential energy (*energy*) and averaged atomic forces (*force*). *force_max* means the maximum of the force uncertainty among different atoms.

3) __uncert_type__ (str): *absolute* (default), *relative*

- Type of the uncertainty. *absolute* and *relative* mean the absolute and relative values of uncertainties, respectively.

4) __uncert_shift__ (float): 2.0 (default)

- A shift of erf function at the middle point for the uncertainty sampling probability function. 1.0 corresponds to a standard deviation of the target uncertainty.

5) __uncer_grad__ (float): 1.0 (default)

- A gradient of erf function at the middle point for the uncertainty sampling probability function, determining the softness of sampling criteria.

6) __output_format__ (str): *trajectory.son* (default), *aims.out*

- Type of FHI-vibes output to be read.

7) __MLIP__ (str): *nequip* (default), *so3krates*

- Type of MLIP models.

<br>
---
### [Active learning parameters]
1) __nstep__ (int): 1 (default)

- The number of subsampling sets.

2) __nmodel__ (int): 1 (default)

- The number of ensemble model sets with different random initializations.

3) __ntrain_init__ (int): 25 (default)

- The number of training data for the initial MLIP models.

4) __ntrain__ (int): 25 (default)

- The number of training data added for each iterative step.

5) __nperiod__ (int): 200 (default)

- The number of MLIP-MD steps for the exploration during the active learning scheme.

<br>
---
### [Molecular dynamics setting]
1) __ensemble__ (str): *NVTLangevin*

- The type of molecular dynamics (MD) ensembles.

2) __temperature__ (float): 300 (default)

- The desired temperature in units of Kelvin (K).

3) __timestep__ (float): 1 (default)

- MD timestep in units of femto-second (fs).

4) __loginterval__ (int): 1 (default)

- The step interval for printing and saving MD steps.

5) __friction__ (float): 0.03 (default)

- The strength of the friction parameter in NVTLangevin ensemble.

<br>
---
### [NequIP setting]
Only four parameters are assigned by the ALmoMD. The other setting is included in *nequip.yaml*.

1) __rmax__ (float): 3.5 (default)

- The cutoff radius in the NequIP.

2) __lmax__ (int): 2 (default)

- The maximum angular mommentum in the NequIP.

3) __nfeatures__ (int): 16 (default)

- The number of features used in the NequIP.

4) __E_gs__ (float): 0.0 (default)

- The reference total energy in units of eV/unitcell to shift the total energies. Recommend to use the total energy of the ground state structure.

<br>
---
### [So3krates setting]
All parameters are employed. For examples,

- r_cut = 5.0
- loss\_variance\_scaling = '--loss-variance-scaling'

Please check the [details](https://github.com/sirmarcel/glp) in the So3krates website. Additionally, the refence energy, __E_gs__, needs to be assigned.

1) __E_gs__ (float): 0.0 (default)

- The reference total energy in units of eV/unitcell to shift the total energies. Recommend to use the total energy of the ground state structure.

<br>
---
# Contents
- [Back to Home](https://keysongkang.github.io/ALmoMD/)
- [Installation Guide](installation.md)
- [User Manuals](documentation.md) ([Input Files](doc_input_file.md), [Input Parameters](doc_input_para.md))
- [Tutorials](../tutorial/tutorial.md)