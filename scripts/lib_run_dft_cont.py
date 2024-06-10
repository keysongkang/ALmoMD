import sys

from ase.io import read as atoms_read

from libs.lib_dft   import run_DFT
from libs.lib_util  import output_init, mpi_print, check_mkdir
from libs.lib_load_model  import load_model
from libs.lib_criteria    import get_result
from libs.lib_mainloop_new    import MLMD_main, MLMD_random
from libs.lib_progress    import check_progress, check_progress_rand, check_index
from libs.lib_termination     import termination

def run_dft_cont(inputs):
    """Function [run_dft_cont]
    Run and continue the ALMD calculation.
    """

    output_init('cont', inputs.version, inputs.rank)
    mpi_print(f'[cont]\tContinue from the previous step (Mode: {inputs.calc_type})', inputs.rank)
    inputs.comm.Barrier()

    # Termination check
    mpi_print(f'[cont]\tConvergence check', inputs.rank)
    signal = termination(inputs)
    if signal == 1:
        mpi_print(f'[cont]\t!!!!Have a nice day. Terminate the code.', inputs.rank)
        sys.exit()

    ## Prepare the ground state structure
    # Read the ground state structure with the primitive cell
    struc_init = atoms_read('geometry.in.supercell', format='aims')
    inputs.NumAtoms = len(struc_init)
    mpi_print(f'[cont]\tRead the reference structure: geometry.in.supercell', inputs.rank)
    inputs.comm.Barrier()

    ### Initizlization step
    ##!! We need to check whether this one is needed or not.
    # Get total_index to resume the MLMD calculation
    if inputs.rank == 0:
        # Open the result.txt file
        inputs.index = check_index(inputs, 'cont')
    inputs.index = inputs.comm.bcast(inputs.index, root=0)
    inputs.comm.Barrier()


    mpi_print(f'[cont]\tLoad trained models: {inputs.temperature}K-{inputs.pressure}bar_{inputs.index}', inputs.rank)
    inputs = load_model(inputs)

    mpi_print(f'[cont]\tProgress check and get validation errors', inputs.rank)
    MD_index, MD_step_index, signal = None, None, None

    # Retrieve the calculation index (MD_index: MLMD_main, signal: termination)
    # to resume the MLMD calculation if a previous one exists.
    ## For active learning sampling,
    if inputs.calc_type == 'active' or inputs.calc_type == 'period':
        # Open the uncertainty output file
        MD_index, MD_step_index, inputs.index, signal = check_progress(inputs, 'cont')
    ## For random sampling,
    elif inputs.calc_type == 'random':
        # Open the uncertainty output file
        MD_index, inputs.index, signal = check_progress_rand(inputs, 'cont')
    else:
        sys.exit("You need to assign calc_type.")

    MD_index = inputs.comm.bcast(MD_index, root=0)
    MD_step_index = inputs.comm.bcast(MD_step_index, root=0)
    inputs.index = inputs.comm.bcast(inputs.index, root=0)
    signal = inputs.comm.bcast(signal, root=0)


    # If we get the signal from check_progress, the script will be terminated.
    if signal == 1:
        mpi_print(f'[cont]\tCalculation is terminated during the check_progress', inputs.rank)
        sys.exit()

    mpi_print(f'[cont]\tCurrent iteration index: {inputs.index}', inputs.rank)
    # Get the total number of traning and validating data at current step
    total_ntrain = inputs.ntrain * inputs.index + inputs.ntrain_init
    total_nval = inputs.nval * inputs.index + inputs.nval_init
    inputs.comm.Barrier()


    ### Get calculators
    # Set the path to folders finding the trained model from NequIP
    workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'
    # Create these folders
    if inputs.rank == 0:
        check_mkdir('MODEL')
        check_mkdir(workpath)
    inputs.comm.Barrier()

    # Currently E_ref uses E_GS which menas zero.
    E_ref = 0.0

    mpi_print(f'[cont]\tImplement MD for active learning (Mode: {inputs.calc_type})', inputs.rank)
    ### MLMD steps
    # For active learning sampling,
    if inputs.calc_type == 'active' or inputs.calc_type == 'period':
        # Run MLMD calculation with active learning sampling for a finite period
        MLMD_main(inputs, MD_index, MD_step_index, inputs.calc_MLIP, E_ref)

    # For random sampling,
    elif inputs.calc_type == 'random':
        # Run MLMD calculation with random sampling
        MLMD_random(inputs, MD_index, inputs.steps_random*inputs.loginterval, inputs.calc_MLIP, E_ref)
    else:
        raise ValueError("[cont]\tInvalid calc_type. Supported values are 'active' and 'random'.")
    inputs.comm.Barrier()

    if inputs.calc_type == 'active' or inputs.calc_type == 'period':
        mpi_print(f'[cont]\tRecord the results: result.txt', inputs.rank)
        # Record uncertainty results at the current step
        get_result(inputs, 'cont')
    inputs.comm.Barrier()

    mpi_print(f'[cont]\tSubmit the DFT calculations for sampled configurations', inputs.rank)
    # Submit job-scripts for DFT calculationsions with sampled configurations and job-dependence for run_dft_gen
    if inputs.rank == 0:
        run_DFT(inputs)
    inputs.comm.Barrier()

    # Submit a job-dependence to execute run_dft_gen after the DFT calculations
    # if inputs.rank == 0:
    #     job_dependency('gen', inputs.num_calc)
    # inputs.comm.Barrier()

    mpi_print(f'[cont]\t!! Finish the MD investigation: Iteration {inputs.index}', inputs.rank)
