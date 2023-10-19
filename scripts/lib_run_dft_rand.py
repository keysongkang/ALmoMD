import son

from libs.lib_util  import output_init, mpi_print, check_mkdir
from libs.lib_npz   import generate_npz_DFT_rand_init, generate_npz_DFT_rand
from libs.lib_train import execute_train_job

import torch
torch.set_default_dtype(torch.float64)


def run_dft_rand(inputs):
    """Function [run_dft_rand]
    Implement the ALMD calculation using random samplings.
    """

    # Print the head
    output_init('rand', inputs.version, inputs.rank)
    mpi_print(f'[rand]\tInitiate the random sampling process', inputs.rank)

    # Read aiMD trajectory file of training data
    metadata, traj = son.load('trajectory_train.son')
    mpi_print(f'[rand]\tRead the initial trajectory data: trajectory_train.son', inputs.rank)
    inputs.comm.Barrier()

    # As it is an initialization step,
    # the total number of training and validation data matches the initial settings
    total_ntrain = inputs.ntrain_init
    total_nval = inputs.nval_init

    # Start from the first step
    inputs.index = 0
    # Set the path to folders storing the training data for NequIP
    workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'

    if inputs.rank == 0:
        # Create these folders
        check_mkdir('MODEL')
        check_mkdir(workpath)

        # Generate first set of training data in npz files from trajectory file
        traj_idx = generate_npz_DFT_rand_init(
            inputs, traj, total_ntrain, total_nval, workpath
        )
    mpi_print(f'[rand]\tGenerate the training data from trajectory_train.son (# of data: {total_ntrain+total_nval})', inputs.rank)
    inputs.comm.Barrier()

    # Training process: Run NequIP
    execute_train_job(inputs, total_ntrain, total_nval, workpath)
    mpi_print(f'[rand]\tSubmit the NequIP training processes for iteration {inputs.index}', inputs.rank)
    inputs.comm.Barrier()

    # Run steps until random_index (which is assigned in input.in)
    while inputs.index < inputs.random_index:
        # Go to the next step
        inputs.index += 1

        # Add the number of new training and validating data
        total_ntrain += inputs.ntrain
        total_nval += inputs.nval

        # Set the path to folders storing the training data for NequIP at the current step
        workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'

        if inputs.rank == 0:
            # Create these folders
            check_mkdir(workpath)

            # Generate first set of training data in npz files from trajectory file
            traj_idx = generate_npz_DFT_rand(inputs, traj, workpath, traj_idx)
        inputs.comm.Barrier()
        mpi_print(f'[rand]\tGenerate the training data from trajectory_train.son (# of data: {total_ntrain+total_nval})', inputs.rank)

        # Training process: Run NequIP
        execute_train_job(inputs, total_ntrain, total_nval, workpath)
        inputs.comm.Barrier()
        mpi_print(f'[rand]\tSubmit the NequIP training processes for iteration {inputs.index}', inputs.rank)

    mpi_print(f'[rand]\t!! Finish the random sampling process', inputs.rank)
