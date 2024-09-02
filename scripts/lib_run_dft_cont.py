import sys

from ase.io import read as atoms_read

from libs.lib_dft   import run_DFT
from libs.lib_util  import output_init, single_print, check_mkdir, get_E_ref
from libs.lib_load_model  import load_model
from libs.lib_criteria    import get_result
from libs.lib_mainloop_new    import MLMD_main, MLMD_random
from libs.lib_progress    import check_progress, check_progress_rand, check_index
from libs.lib_termination     import termination

def run_dft_cont(inputs):
    """Function [run_dft_cont]
    Run and continue the ALMD calculation.
    """

    output_init('cont', inputs.version)
    single_print(f'[cont]\tContinue from the previous step (Mode: {inputs.calc_type})')

    # Termination check
    single_print(f'[cont]\tConvergence check')
    signal = termination(inputs)
    if signal == 1:
        single_print(f'[cont]\t!!!!Have a nice day. Terminate the code.')
        sys.exit()

    ## Prepare the ground state structure
    # Read the ground state structure with the primitive cell
    struc_init = atoms_read('geometry.in.supercell', format='aims')
    inputs.NumAtoms = len(struc_init)
    single_print(f'[cont]\tRead the reference structure: geometry.in.supercell')

    ### Initizlization step
    ##!! We need to check whether this one is needed or not.
    # Get total_index to resume the MLMD calculation
    # Open the result.txt file
    inputs.index = check_index(inputs, 'cont')

    single_print(f'[cont]\tLoad trained models: {inputs.temperature}K-{inputs.pressure}bar_{inputs.index}')
    inputs = load_model(inputs)

    single_print(f'[cont]\tProgress check and get validation errors')
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

    # If we get the signal from check_progress, the script will be terminated.
    if signal == 1:
        single_print(f'[cont]\tCalculation is terminated during the check_progress')
        sys.exit()

    if inputs.idx_atom == 'random':
        if MD_step_index != 0:
            import re
            with open('almomd.out', 'r') as file:
                content = file.read()

            matches = re.findall(r'Randomly\s+selected\s+biased\s+idx_atom\s*:\s*(\d+)', content)
            inputs.idx_atom = int(matches[-1])
        else:
            import random
            inputs.idx_atom = random.randint(0, inputs.NumAtoms)
            single_print(f'[cont]\tRandomly selected biased idx_atom : {inputs.idx_atom}')
    else:
        inputs.idx_atom = int(inputs.idx_atom)

    single_print(f'[cont]\tCurrent iteration index: {inputs.index}')
    # Get the total number of traning and validating data at current step
    total_ntrain = inputs.ntrain * inputs.index + inputs.ntrain_init
    total_nval = inputs.nval * inputs.index + inputs.nval_init

    ### Get calculators
    # Set the path to folders finding the trained model from NequIP
    workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'
    # Create these folders
    check_mkdir('MODEL')
    check_mkdir(workpath)

    # Currently E_ref uses E_GS which menas zero.
    E_ref = get_E_ref(inputs.nmodel, inputs.nstep, inputs.calc_MLIP)

    single_print(f'[cont]\tImplement MD for active learning (Mode: {inputs.calc_type})')
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

    if inputs.calc_type == 'active' or inputs.calc_type == 'period':
        single_print(f'[cont]\tRecord the results: result.txt')
        # Record uncertainty results at the current step
        get_result(inputs, 'cont')

    single_print(f'[cont]\tSubmit the DFT calculations for sampled configurations')
    # Submit job-scripts for DFT calculationsions with sampled configurations and job-dependence for run_dft_gen
    
    run_DFT(inputs)

    # Submit a job-dependence to execute run_dft_gen after the DFT calculations
    # if inputs.rank == 0:
    #     job_dependency('gen', inputs.num_calc)
    # inputs.comm.Barrier()

    single_print(f'[cont]\t!! Finish the MD investigation: Iteration {inputs.index}')
