import sys

from libs.lib_util  import output_init, single_print, check_mkdir
from libs.lib_npz   import generate_npz_DFT
from libs.lib_train import execute_train_job
from libs.lib_progress    import check_progress, check_progress_rand, check_index


def run_dft_gen(inputs):
    """Function [run_dft_gen]
    Extract DFT results, generate the training data, and execute NequIP.
    """
    output_init('gen', inputs.version)
    single_print('')

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
        single_print('')
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
    single_print('')
    execute_train_job(inputs, total_ntrain, total_nval, workpath)

    # Auto-chaining disabled by default. Uncomment to have almomd submit
    # the next job-cont.slurm with an sbatch dependency on the training
    # jobs just submitted above. Only useful when the auto-submissions in
    # lib_dft.run_DFT and lib_train.execute_train_job are also enabled.
    # job_dependency('cont', inputs.num_mdl_calc)

    single_print(f'[gen]\t!! Finish the training data generation: Iteration {inputs.index}')
