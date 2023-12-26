import sys

from ase.io import read as atoms_read

from libs.lib_util  import output_init, mpi_print, check_mkdir, job_dependency
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
    output_init('gen', inputs.version, inputs.rank)
    mpi_print(f'[gen]\tGenerate the NequIP inputs from DFT results', inputs.rank)
    inputs.comm.Barrier()

    ## Prepare the ground state structure
    # Read the ground state structure with the primitive cell
    struc_init = atoms_read('geometry.in.supercell', format='aims')

    mpi_print(f'[gen]\tRead the reference structure: geometry.in.supercell', inputs.rank)
    inputs.comm.Barrier()

    ### Initizlization step
    ##!! We need to check whether this one is needed or not.
    # Get total_index to resume the MLMD calculation
    if inputs.rank == 0:
        inputs.index = check_index(inputs, 'gen')
    inputs.index = inputs.comm.bcast(inputs.index, root=0)

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

    MD_index = inputs.comm.bcast(MD_index, root=0)
    MD_step_index = inputs.comm.bcast(MD_step_index, root=0)
    inputs.index = inputs.comm.bcast(inputs.index, root=0)
    signal = inputs.comm.bcast(signal, root=0)


    # If we get the signal from check_progress, the script will be terminated.
    if signal == 1:
        mpi_print(f'[gen]\tCalculation is terminated during the check_progress', inputs.rank)
        sys.exit()
    inputs.comm.Barrier()

    mpi_print(f'[gen]\tCurrent iteration index: {inputs.index}', inputs.rank)
    # Get the total number of traning and validating data at current step
    total_ntrain = inputs.ntrain * inputs.index + inputs.ntrain_init
    total_nval = inputs.nval * inputs.index + inputs.nval_init


    ### Get DFT results and generate training data
    # Set the path to folders storing the training data for NequIP
    workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'
    if inputs.rank == 0:
        check_mkdir('MODEL')
        check_mkdir(workpath)
    inputs.comm.Barrier()

    mpi_print(f'[gen]\tGenerate the training data from DFT results (# of data: {total_ntrain+total_nval})', inputs.rank)
    # Generate first set of training data in npz files from trajectory file
    if inputs.rank == 0:
        generate_npz_DFT(inputs, workpath)
    inputs.comm.Barrier()

    # Training process: Run NequIP
    mpi_print(f'[gen]\tSubmit the NequIP training processes', inputs.rank)
    execute_train_job(inputs, total_ntrain, total_nval, workpath)
    inputs.comm.Barrier()

    # Submit a job-dependence to execute run_dft_cont after the NequIP training
    # mpi_print(f'[gen]\tSubmit a job for cont with dependency', inputs.rank)
    # if inputs.rank == 0:
    #     job_dependency('cont', inputs.num_mdl_calc)
    # inputs.comm.Barrier()

    mpi_print(f'[gen]\t!! Finish the training data generation: Iteration {inputs.index}', inputs.rank)
