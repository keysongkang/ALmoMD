import os
from vibes import son
import numpy as np

from libs.lib_util  import output_init, single_print, check_mkdir
from libs.lib_npz   import generate_npz_DFT_rand_init, generate_npz_DFT_rand
from libs.lib_train import execute_train_job
from libs.lib_progress    import check_index

import torch
torch.set_default_dtype(torch.float64)


def run_dft_rand(inputs):
    """Function [run_dft_rand]
    Implement the ALMD calculation using random samplings.
    """

    # Print the head
    output_init('rand', inputs.version)
    single_print(f'[rand]\tInitiate the random sampling process')

    # Read aiMD trajectory file of training data
    metadata, traj = son.load('trajectory_train.son')
    single_print(f'[rand]\tRead the initial trajectory data: trajectory_train.son')

    # As it is an initialization step,
    # the total number of training and validation data matches the initial settings
    total_ntrain = inputs.ntrain_init
    total_nval = inputs.nval_init

    ### Initizlization step
    ##!! We need to check whether this one is needed or not.
    # Get total_index to resume the MLMD calculation
    if os.path.exists(f'MODEL/{inputs.temperature}K-{inputs.pressure}bar_0'):
        # Open the result.txt file
        inputs.index = check_index(inputs, 'dft_rand')
    else:
        inputs.index = 0

    single_print(f'[rand]\tIdentify the index {inputs.index}')

    # Start from the first step
    if inputs.index == 0:
        # Set the path to folders storing the training data for NequIP
        workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'

        # Create these folders
        check_mkdir('MODEL')
        check_mkdir(workpath)

        # Generate first set of training data in npz files from trajectory file
        traj_idx = generate_npz_DFT_rand_init(
            inputs, traj, total_ntrain, total_nval, workpath
        )
        single_print(f'[rand]\tGenerate the training data from trajectory_train.son (# of data: {total_ntrain+total_nval})')

        # Training process: Run NequIP
        execute_train_job(inputs, total_ntrain, total_nval, workpath)
        single_print(f'[rand]\tSubmit the NequIP training processes for iteration {inputs.index}')

        # Go to the next step
        inputs.index += 1
    else:
        single_print(f'[rand]\tExtract index of training data')
        traj_idx = []
        for idx in range(inputs.index):
            for jdx in range(inputs.nstep):
                data = np.load(f'MODEL/{inputs.temperature}K-{inputs.pressure}bar_{idx}/data-train_{jdx}.npz')
                
                for item, jtem in zip(data['E'], data['F']):
                    for ktem, ltem in enumerate(traj):
                        if ltem['calculator']['energy'] - inputs.E_gs == item and ltem['calculator']['forces'][0][0] == jtem[0][0]:
                            traj_idx.append(ktem)

    # Run steps until random_index (which is assigned in input.in)
    while inputs.index < inputs.random_index:

        # Add the number of new training and validating data
        total_ntrain += inputs.ntrain
        total_nval += inputs.nval

        # Set the path to folders storing the training data for NequIP at the current step
        workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'

        # Create these folders
        check_mkdir(workpath)

        # Generate first set of training data in npz files from trajectory file
        traj_idx = generate_npz_DFT_rand(inputs, traj, workpath, traj_idx)
        single_print(f'[rand]\tGenerate the training data from trajectory_train.son (# of data: {total_ntrain+total_nval})')

        # Training process: Run NequIP
        execute_train_job(inputs, total_ntrain, total_nval, workpath)
        single_print(f'[rand]\tSubmit the NequIP training processes for iteration {inputs.index}')

        # Go to the next step
        inputs.index += 1

    single_print(f'[rand]\t!! Finish the random sampling process')

