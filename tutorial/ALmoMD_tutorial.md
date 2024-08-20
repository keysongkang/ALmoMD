#ALmoMD Tutorial

## Installation
1) Prepare a conda environment

```
conda create -n almomd python=3.9
conda activate almomd
```

2) Install the [FHI-vibes](https://vibes-developers.gitlab.io/vibes/Installation/). Follow the instructions on the linked website.

3) Install the [NequIP](https://github.com/mir-group/nequip). Follow the instructions on the linked website.

4) Install the ALmoMD.

```
git clone https://github.com/keysongkang/ALmoMD.git
cd almomd
pip install .
```


## Input files

1) __trajectory.son__: An aiMD trajectory file from the FHI-vibes. This will be used to obtain training and testing data.

2) __geometry.in.supercell__: A supercell with the ground-state atomic structure. This should have the same-size structure that you plan to use for training data.

3) __nequip.yaml__: A NequIP input file. You need to modify this file depending on your system. __*r_max*__, __*l_max*__, and __*n_features*__ are controlled by __input.in__. Particularly, you need to check __*chemical_symbols*__ and __*loss_coeffs*__.

```YAML
dataset_seed: 0              # data set seed
append: true                 # set true if a restarted run should append to the previous log file
default_dtype: float64       # type of float to use, e.g. float32 and float64
allow_tf32: false            # whether to use TensorFloat32 if it is available
device:  cuda                # which device to use. Default: automatically detected cuda or "cpu"

# network
model_builders:
  - SimpleIrrepsConfig
  - EnergyModel
  - PerSpeciesRescale
  - ForceOutput
  - RescaleEnergyEtc

num_layers: 4                # number of interaction blocks, we find 3-5 to work best
parity: true                 # whether to include features with odd mirror parity; often turning parity off gives equally good results but faster networks, so do consider this
nonlinearity_type: gate      # may be 'gate' or 'norm', 'gate' is recommended

# alternatively, the irreps of the features in various parts of the network can be specified directly:
# the following options use e3nn irreps notation
# either these four options, or the above three options, should be provided--- they cannot be mixed.

# radial network basis
num_basis: 8                 # number of basis functions used in the radial basis, 8 usually works best
BesselBasis_trainable: true  # set true to train the bessel weights
PolynomialCutoff_p: 6        # p-value used in polynomial cutoff function
invariant_layers: 2          # number of radial layers, we found it important to keep this small, 1 or 2
invariant_neurons: 64        # number of hidden neurons in radial function, again keep this small for MD applications, 8 - 32, smaller is faster
avg_num_neighbors: auto      # number of neighbors to divide by, None => no normalization.
use_sc: true                 # use self-connection or not, usually gives big improvement

# data set
# the keys used need to be stated at least once in key_mapping, npz_fixed_field_keys or npz_keys
# key_mapping is used to map the key in the npz file to the NequIP default values (see data/_key.py)
# all arrays are expected to have the shape of (nframe, natom, ?) except the fixed fields
dataset: npz                 # type of data set, can be npz or ase
key_mapping:
  z: atomic_numbers          # atomic species, integers
  E: total_energy            # total potential eneriges to train to
  F: forces                  # atomic forces to train to
  R: pos                     # raw atomic positions
  unit_cell: cell
  pbc: pbc
chemical_symbols:
  - Cu
  - I

verbose: info                # the same as python logging, e.g. warning, info, debug, error; case insensitive
log_batch_freq: 100          # batch frequency, how often to print training errors withinin the same epoch
log_epoch_freq: 1            # epoch frequency, how often to print 
save_checkpoint_freq: -1     # frequency to save the intermediate checkpoint. no saving of intermediate checkpoints when the value is not positive.
save_ema_checkpoint_freq: -1 # frequency to save the intermediate ema checkpoint. no saving of intermediate checkpoints when the value is not positive.
# scalar nonlinearities to use — available options are silu, ssp (shifted softplus), tanh, and abs.
# Different nonlinearities are specified for e (even) and o (odd) parity;
# note that only tanh and abs are correct for o (odd parity).
nonlinearity_scalars:
  e: silu
  o: tanh

nonlinearity_gates:
  e: silu
  o: tanh

# training
learning_rate: 0.01          # learning rate, we found 0.01 to work best - this is often one of the most important hyperparameters to tune
batch_size: 1                # batch size, we found it important to keep this small for most applications 
max_epochs: 1000000          # stop training after _ number of epochs
train_val_split: random      # can be random or sequential. if sequential, first n_train elements are training, next n_val are val, else random 
shuffle: true                # If true, the data loader will shuffle the data
metrics_key: validation_loss # metrics used for scheduling and saving best model. Options: loss, or anything that appears in the validation batch step header, such as f_mae, f_rmse, e_mae, e_rmse
use_ema: True                # if true, use exponential moving average on weights for val/test
ema_decay: 0.999             # ema weight, commonly set to 0.999
ema_use_num_updates: True    # whether to use number of updates when computing averages
# loss function
loss_coeffs:                 # different weights to use in a weighted loss functions
 forces: 100                 # for MD applications, we recommed a force weight of 100 and an energy weight of 1
 total_energy:               # alternatively, if energies are not of importance, a force weight 1 and an energy weight of 0 also works.
    - 1
# output metrics

# early stopping based on metrics values. 
# LR, wall and any keys printed in the log file can be used. 
# The key can start with Training or validation. If not defined, the validation value will be used.
early_stopping_patiences:    # stop early if a metric value stopped decreasing for n epochs
  validation_loss: 25

early_stopping_delta:        # If delta is defined, a decrease smaller than delta will not be considered as a decrease
  validation_loss: 0.005

early_stopping_cumulative_delta: false # If True, the minimum value recorded will not be updated when the decrease is smaller than delta

early_stopping_lower_bounds: # stop early if a metric value is lower than the bound
  LR: 1.0e-5

early_stopping_upper_bounds: # stop early if a metric value is higher than the bound
  wall: 1.0e+100

metrics_components:
  - - forces                       # key
    - rmse                         # "rmse" or "mse"
    - report_per_component: True   # if true, statistics on each component (i.e. fx, fy, fz) will be counted separately
  - - forces
    - mae
    - report_per_component: True
  - - total_energy
    - mae
    - PerAtom: True                # if true, energy is normalized by the number of atoms
  - - total_energy
    - rmse
    - PerAtom: True
      
# the name `optimizer_name`is case sensitive
optimizer_name: Adam               # default optimizer is Adam in the amsgrad mode
optimizer_amsgrad: true
optimizer_betas: !!python/tuple
  - 0.9
  - 0.999
optimizer_eps: 1.0e-08
optimizer_weight_decay: 0

# lr scheduler
lr_scheduler_name: ReduceLROnPlateau
lr_scheduler_patience: 50
lr_scheduler_factor: 0.5
```

4) __job-cont.slurm__: A job script for the MLIP-MD.

```Bash
(This is an example for the COBRA system.)
#!/bin/bash -l

#SBATCH -J test_gpu
#SBATCH -o ./out.%j
#SBATCH -e ./err.%j
#SBATCH -D ./
#SBATCH --ntasks=1         # launch job on a single core
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1  #   on a shared node
#SBATCH --mem=92500       # memory limit for the job
#SBATCH --time=01:00:00

(Load your modules)
conda activate almomd

srun almomd cont >> almomd.out
```

5) __job-nequip-gpu.slurm__: A job script for training the NequIP models. This script should be __*empty*__.

```Bash
(This is an example for the COBRA system.)
#!/bin/bash -l

#SBATCH -J test_gpu
#SBATCH -o ./out.%j
#SBATCH -e ./err.%j
#SBATCH -D ./
#SBATCH --ntasks=1         # launch job on a single core
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=1  #   on a shared node
#SBATCH --mem=92500       # memory limit for the job
#SBATCH --time=01:00:00

(Load your modules)
conda activate almomd

(Just leave here empty)
```

6) __input.in__: An input file for the ALmoMD code.

```
#[Active learning types]
calc_type     : active             # (active) sampling up to the number of training data needed (period) sampling during the assigned period
al_type       : force_max          # Uncertainty type: force_max (Maximum atomic force uncertainty)
uncert_type   : absolute           # Absolute or relative uncertainty (absolute / relative)
output_format : trajectory.son     # File format of the output file (trajectory.son / aims.out / nequip)
device        : cuda               # Calating core type (cpu / cuda)

#[Uncertainty sampling criteria]
uncert_shift  : 2.0                # How far from the average? 1.0 means one standard deviation
uncert_grad   : 1.0                # Gradient near boundary

#[Active learning setting]
nstep         : 3                  # The number of models using the subsampling
nmodel        : 2                  # The number of models with different random initialization
ntrain_init   : 25                 # The number of training data for the initial NequIP model
ntrain        : 25                 # The number of newly added training data for each AL step

#[Molecular dynamics setting]
ensemble      : NVTLangevin        # Currently, only NVT Langevin MD is available
temperature   : 300
timestep      : 5
loginterval   : 1
friction      : 0.03

#[NequIP setting]
rmax          : 5.0 
lmax          : 3
nfeatures     : 32
num_calc      : 16                 # The number of job scripts for DFT calculations
num_mdl_calc  : 6                  # The number of job scripts for MLIP training calculations
E_gs          : -1378450.95449287  # The reference potential energy (Energy of the geometry.in.supercell)
```

7) __DFT_INPUTS__: A directory for the DFT inputs

__aims.in__: A input for the FHI-vibes single-point calculation. You need to make sure that your calculation reproduces the aiMD energy and force appropriately. This is very tricky since some old FHI-aims calculations have different environments, which requires very careful check for inputs.

__job-vibes.slurm__: A job script for the FHI-vibes. This script should be __*empty*__, too.

```Bash
(This is an example for the COBRA system.)
#!/bin/bash -l

#SBATCH -o ./out_o.%j
#SBATCH -e ./out_e.%j
#SBATCH -D ./                  # Initial working directory
#SBATCH -J test_slurm          # Job Name
#
#SBATCH --nodes=1              # Number of nodes and MPI tasks per node
#SBATCH --ntasks-per-node=40
#
#SBATCH --mail-type=none
#SBATCH --mail-user=kang@fhi-berlin.mpg.de
#
#SBATCH --time=01:40:00        # Wall clock limit (max. is 24 hours)

(Load your modules)
conda activate almomd (or your conda environment for FHI-vibes)

(Just leave here empty)
```

## Initialization

We are going to have a tutorial of the ALmoMD with an example of CuI.

### Ground truth
But the actual implementation of ALmoMD with DFT is too demanding for the tutorial purpose. Therefore, in this tutorial, we are going to have a pretrained MLIP model as a ground truth instead of DFT. To do that, you need __REFER__ instead of __DFT_INPUTS__.

__REFER__: A directory containing the pretrained MLIP depolyed model (deployed-model\_0\_0.pth). 

Accodingly, your ALmoMD inputs should be modified as below.

```
output_format : nequip             # Make the pretrained NequIP model as the ground truth
E_gs          : -0.142613646819514 # New corresponding reference potential energy (Energy of the geometry.in.supercell)
```

### Train initial MLIP models
1) Split the aiMD trajectories into training and testing data. This can be implemented by a command of `almomd utils split (# of testing data) (E_gs)`. In this practice, we will use 100 testing data.

```
almomd utils split 100 -0.142613646819514
```

It will create two files __trajectory_train.son__, __trajectory_test.son__, and a directory __MODEL__ containing __data_test.npz__.

2) Create the training inputs for initial MLIP modes.

```
almomd init
```

It will create a directory of __300K-0bar\_0__ inside of __MODEL__. 

```
cd MODEL/300-bar_0
```

You will find 3 training data (__data-train\_\*.npz__), and 6 NequIP inputs (__input\_\*\_\*.yaml__) and corresponding job scripts (__job-nequip-gpu
\_\*\_\*.slurm__). This is because you assign 3 subsampling and 2 random initialization in __input.in__, leading to a total of 6 (=2*3) different MLIP models.

3) Submit your job scripts to train MLIP models.

```
sbatch job-nequip-gpu\_0.slurm; sbatch job-nequip-gpu\_1.slurm; sbatch job-nequip-gpu\_2.slurm; sbatch job-nequip-gpu\_3.slurm; sbatch job-nequip-gpu\_4.slurm; sbatch job-nequip-gpu\_5.slurm
```

4) When your training is done, you will get deployed MLIP models (__depolyed-model\_\*\_\*.pth__).


## Active Learning Procedure
The active learning iterative loop in the ALmoMD consists of three major steps (MLIP exploration, DFT calculation, and MLIP training).

### MLIP exploration
When you have MLIP models, the ALmoMD will explore the configurational space via MLIP-MD. This can be conducted by submit your __job-cont.slurm__.

```
sbatch job-cont.slurm
```

It will generate many files and directories. But, __almomd.out__, __result.txt__, and __UNCERT/uncertainty-300K-0bar\_*.txt__ are important files that users know.

1) __almomd.out__: It shows the overall process of the ALmoMD.
   
2) __result.txt__: It contains the testing results and their MLIP uncertainty at each active learning step.

3) __UNCERT/uncertainty-300K-0bar\_*.txt__: It records the result of the MLIP-MD steps. You can recognize which MD snapshots are sampled.

When it samples all data, it will create a directory of __CALC/300K-0bar\_\*__, where all DFT inputs for the sampled snapshots are prepared.

### DFT calculation
In each iteration, you need to go into the most recent __CALC/300K-0bar\_\*__.

```
cd CALC/300-0bar_1
```

You need to submit all job scripts.

```
sbatch job-vibes_0.slurm; sbatch job-vibes_1.slurm; sbatch job-vibes_2.slurm; sbatch job-vibes_3.slurm; sbatch job-vibes_4.slurm; sbatch job-vibes_5.slurm; sbatch job-vibes_6.slurm; sbatch job-vibes_7.slurm; sbatch job-vibes_8.slurm; sbatch job-vibes_9.slurm; sbatch job-vibes_10.slurm; sbatch job-vibes_11.slurm; sbatch job-vibes_12.slurm; sbatch job-vibes_13.slurm; sbatch job-vibes_14.slurm; sbatch job-vibes_15.slurm
```

### MLIP training
Once all DFT calculations are finished, go back to main directory where __result.txt__ exists.

```
almomd gen
```

This will add all new DFT outcomes into the training data. The new training data, inputs, and corresponding job scripts are generated in the most recent __MODEL/300K-0bar\_\*__. Then, submit all job scripts.

```
sbatch job-nequip-gpu\_0.slurm; sbatch job-nequip-gpu\_1.slurm; sbatch job-nequip-gpu\_2.slurm; sbatch job-nequip-gpu\_3.slurm; sbatch job-nequip-gpu\_4.slurm; sbatch job-nequip-gpu\_5.slurm
```

When your training is done, you will get deployed MLIP models (__depolyed-model\_\*\_\*.pth__). Then, go back to __MLIP exploration__ section to complete the loop.
