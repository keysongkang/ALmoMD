# Installation
ALmoMD requires a Python environment (>=3.7, tested with Python 3.9) with Message Passing Interface (MPI) settings implemented using mpi4py.
NequIP (>=0.5.6) necessitates the implementation of PyTorch modules on GPU nodes (Check [details](https://github.com/mir-group/nequip)).
FHI-aims (newer than 200112_2) mandates a Fortran compiler with MPI settings (Check [FHI-aims](https://fhi-aims-club.gitlab.io/tutorials/basics-of-running-fhi-aims/preparations/) and [FHI-vibes](https://vibes-developers.gitlab.io/vibes/Installation/).

## Step-by-Step Setup
To set up a Python environment, you can manually build your own environment. However, we recommend preparing Conda settings. All the steps are based on the implementation of Conda settings, which can be found here. If you already have your own Python environment, you just need to conduct step 4.


1. To set up a Python environment, you can manually build your own environment, but we recommend preparing Conda settings first. For installation instructions, please refer to [this Conda link](https://docs.conda.io/projects/conda/en/23.1.x/user-guide/install/index.html).

2. Create a Conda environment for building a Python 3.9 environment.
```
conda create -n almomd python=3.9
```

3. Activate the environment. Whenever you want to use this code package, please type this command to activate the environment later.
```
conda activate almomd
```

4. Get the ALmoMD from github and install it.
```
git clone git@github.com:keysongkang/ALmoMD.git
cd ALmoMD
pip install .
```
