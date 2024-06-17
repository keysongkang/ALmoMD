from vibes import son

from libs.lib_util  import output_init, single_print, check_mkdir, job_dependency
from libs.lib_npz   import generate_npz_DFT_init
from libs.lib_train import execute_train_job

import torch
torch.set_default_dtype(torch.float64)


def run_dft_init(inputs):
    """Function [run_dft_init]
    Perform the calculation by initializing with the aiMD trajectory.
    """

    # Print the head
    output_init('init', inputs.version)
    single_print(f'[init]\tInitiate the active learning process')

    # Read aiMD trajectory file of training data
    metadata, traj = son.load('trajectory_train.son')
    single_print(f'[init]\tRead the initial trajectory data: trajectory_train.son')

    # Set the path to folders storing the training data for NequIP
    workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'
    # Create these folders
    check_mkdir('MODEL')
    check_mkdir(workpath)

    # Generate first set of training data in npz files from trajectory file
    generate_npz_DFT_init(inputs, traj, workpath)
    del traj  # Remove it to reduce the memory usage
    single_print(f'[init]\tGenerate the training data (# of data: {inputs.ntrain_init+inputs.nval_init})')

    # Training process: Run NequIP
    execute_train_job(
        inputs, inputs.ntrain_init, inputs.nval_init, workpath
    )
    single_print(f'[init]\tSubmit the NequIP training processes')

    # Submit a job-dependence to execute run_dft_cont after the NequIP training
    # For 'converge' setting, we don't need to submit it.
    # if not inputs.calc_type == 'converge':
    #     if inputs.rank == 0:
    #         job_dependency('cont', inputs.nmodel)

    single_print(f'[init]\t!! Finish the initialization')
