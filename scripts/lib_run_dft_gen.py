import sys

from ase.io import read as atoms_read

from libs.lib_util  import output_init, single_print, check_mkdir, job_dependency
from libs.lib_npz   import generate_npz_DFT
from libs.lib_train import execute_train_job
from libs.lib_load_model  import load_model
from libs.lib_progress    import check_progress, check_progress_rand, check_index

import torch
torch.set_default_dtype(torch.float64)


def run_dft_gen(inputs):
    """Function [run_dft_gen]
    Extract DFT results, generate the training data, and execute NequIP.
    """
    output_init('gen', inputs.version)
    single_print(f'[gen]\tGenerate the NequIP inputs from DFT results')

    ## Prepare the ground state structure
    # Read the ground state structure with the primitive cell
    struc_init = atoms_read('geometry.in.supercell', format='aims')

    single_print(f'[gen]\tRead the reference structure: geometry.in.supercell')

    ### Initizlization step
    ##!! We need to check whether this one is needed or not.
    # Get total_index to resume the MLMD calculation
    inputs.index = check_index(inputs, 'gen')

    # mpi_print(f'[gen]\tLoad trained models: {inputs.temperature}K-{inputs.pressure}bar_{inputs.index}', inputs.rank)
    # inputs = load_model(inputs)

    MD_index, MD_step_index, signal = None, None, None

    # Retrieve the calculation index (MD_index: MLMD_main, signal: termination)
    # to resume the MLMD calculation if a previous one exists.
    ## For active learning sampling,
    if inputs.calc_type == 'active' or inputs.calc_type == 'period':
        # Open the uncertainty output file
        MD_index, MD_step_index, inputs.index, signal = check_progress(inputs, 'gen')
    ## For random sampling,
    elif inputs.calc_type == 'random':
        # Open the uncertainty output file
        MD_index, inputs.index, signal = check_progress_rand(inputs, 'gen')
    else:
        single_print('You need to assign calc_type.')
        signal = 1


    # If we get the signal from check_progress, the script will be terminated.
    if signal == 1:
        single_print(f'[gen]\tCalculation is terminated during the check_progress')
        sys.exit()

    single_print(f'[gen]\tCurrent iteration index: {inputs.index}')
    # Get the total number of traning and validating data at current step
    total_ntrain = inputs.ntrain * inputs.index + inputs.ntrain_init
    total_nval = inputs.nval * inputs.index + inputs.nval_init


    ### Get DFT results and generate training data
    # Set the path to folders storing the training data for NequIP
    workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'
    check_mkdir('MODEL')
    check_mkdir(workpath)

    single_print(f'[gen]\tGenerate the training data from DFT results (# of data: {total_ntrain+total_nval})')
    # Generate first set of training data in npz files from trajectory file
    generate_npz_DFT(inputs, workpath)

    # Training process: Run NequIP
    single_print(f'[gen]\tSubmit the NequIP training processes')
    execute_train_job(inputs, total_ntrain, total_nval, workpath)

    # Submit a job-dependence to execute run_dft_cont after the NequIP training
    # mpi_print(f'[gen]\tSubmit a job for cont with dependency', inputs.rank)
    # if inputs.rank == 0:
    #     job_dependency('cont', inputs.num_mdl_calc)
    # inputs.comm.Barrier()

    single_print(f'[gen]\t!! Finish the training data generation: Iteration {inputs.index}')
