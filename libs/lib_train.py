import os
import numpy as np

from libs.lib_util        import single_print, rm_mkdir

import torch
torch.set_default_dtype(torch.float64)

def nequip_train_job(
    inputs, ntrain, nval, workpath, index_nstep, index_nmodel, dply_model
):
    """Function [nequip_train_job]
    Generate a NequIP input and provide a command line
    to initiate training and deploy the trained model for NequIP

    Parameters:

    ntrain: int
        The number of added training data for each iterative step
    nval: int
        The number of added validating data for each iterative step
    rmax: float
        The rmax input for NequIP
    lmax: int
        The lmax input for NequIP
    nfeatures: int
        The nfeatures input for NequIP
    workpath: str
        Path to the working directory
    index_nstep: int
        The index of current subsampling set
    index_nmodel: int
        The index of current ensemble model set
    dply_model: str
        Path to the deployed model

    Returns:

    script_str: str
        Command lines to be included in the job script
    """

    deploy_model = f'./{dply_model}'
    
    # Check the current path
    currentpath  = os.getcwd()

    # Prepare ingredients for the NequIP input
    with open('./nequip.yaml', 'r') as nequip_initial:
        nequip_input_default = nequip_initial.read()

    # Loss function: 1/varience
    data_nstep = np.load(f'{workpath}/data-train_{index_nstep}.npz')
    var_E = 1/np.var(np.array(data_nstep['E']))
    var_F = 1/np.var(np.array(data_nstep['F']).flatten())

    if inputs.train_stress:
        var_S = 1/np.var(np.array(data_nstep['stress']).flatten())

    nequip_input_extra = (
        f'root: train_{index_nmodel}_{index_nstep}/projects\n'
        + 'run_name: train\n'
        + 'workdir: train\n'
        + f'dataset_file_name: ./data-train_{index_nstep}.npz\n'
        + f'n_train: {ntrain}\n'
        + f'n_val: {nval}\n'
        + f'r_max: {inputs.rmax}\n'
        + f'l_max: {inputs.lmax}\n'
        + f'num_features: {inputs.nfeatures}\n\n'
        + f'seed: {index_nmodel}\n\n'
    )

    if inputs.loss_var:
        nequip_input_extra += (
            'loss_coeffs:\n'
            + f' total_energy: {var_E}\n'
            + f' forces: {var_F}\n'
            )
        if inputs.train_stress:
            nequip_input_extra += f' stress: {var_S}\n'

    job_script_extra = 'srun nequip-train'
    job_script_deploy = 'srun nequip-deploy build --train-dir'
    
    # Move to the working directory
    os.chdir(workpath)

    # Input/Output names
    modelpath    = f'./train_{index_nmodel}_{index_nstep}/projects/train/'
    nequip_input = f'input_{index_nmodel}_{index_nstep}.yaml'

    # Write an input for NequIP
    with open(nequip_input, 'w') as writing_input:
        writing_input.write(nequip_input_extra)
        writing_input.write(nequip_input_default)

    # Move back to the original path
    os.chdir(currentpath)
    
    return f'{job_script_extra} {nequip_input} --warn-unused\n{job_script_deploy} {modelpath} {deploy_model}\n'



def so3krates_train_job(
    inputs, ntrain, nval, workpath, index_nstep, index_nmodel, dply_model
):
    so3krates_input_extra = (
        f'--ckpt-dir {dply_model} '
        + f'--r-cut {inputs.r_cut} '
        + f'--l {inputs.l} '
        + f'--f {inputs.f} '
        + f'--l-min {inputs.l_min} '
        + f'--l-max {inputs.l_max} '
        + f'--max-body-order {inputs.max_body_order} '
        + f'--f-body-order {inputs.f_body_order} '
        + f'-we {inputs.we} '
        + f'-wf {inputs.wf} '
        + f'{inputs.loss_variance_scaling} '
        + f'--epochs {inputs.epochs} '
        + f'--train-split {inputs.train_split} '
        + f'{inputs.mic} '
        + f'{inputs.float64} '
        + f'--lr {inputs.lr} '
        + f'--lr-stop {inputs.lr_stop} '
        + f'--lr-decay-exp-transition-steps {inputs.lr_decay_exp_transition_steps} '
        + f'--lr-decay-exp-decay-factor {inputs.lr_decay_exp_decay_factor} '
        + f'{inputs.shift_mean} '
        + f'--seed-model {index_nmodel} '
        + f'--seed-data {inputs.seed_data} '
        + f'--seed-training {inputs.seed_training} '
        + f'--outfile-inputs inputs_{index_nmodel}_{index_nstep}.json '
        + f'{inputs.overwrite_module} '
        + f'{inputs.ace} '
    )

    if inputs.ws != 'None':
        so3krates_input_extra += f'-ws {inputs.ws} '

    if inputs.eval_energy_t != 'None':
        so3krates_input_extra += f'--eval-every-t {inputs.eval_energy_t} '

    if inputs.clip_by_global_norm != 'None':
        so3krates_input_extra += f'--clip-by-global-norm {inputs.clip_by_global_norm} '
    
    if inputs.size_batch != 'None':
        so3krates_input_extra += f'--size-batch {inputs.size_batch} '

    if inputs.size_batch_training != 'None':
        so3krates_input_extra += f'--size-batch-training {inputs.size_batch_training} '

    if inputs.size_batch_validation != 'None':
        so3krates_input_extra += f'--size-batch-validation {inputs.size_batch_validation} '

    if inputs.wandb_name != 'None':
        so3krates_input_extra += f'--wandb-name {inputs.wandb_name} '

    if inputs.wandb_group != 'None':
        so3krates_input_extra += f'--wandb-group {inputs.wandb_group} '

    if inputs.wandb_project != 'None':
        so3krates_input_extra += f'--wandb-project {inputs.wandb_project} '

    job_script_extra = 'srun sokrates_train'
    so3krates_input = f'data-train_{index_nstep}.npz'

    return f'{job_script_extra} {so3krates_input_extra}{so3krates_input}\n'


def execute_train_job(
    inputs, ntrain, nval, workpath
):
    """Function [execute_train_job]
    Generate a NequIP input and submit the corresponding job script.

    Parameters:

    ntrain: int
        The number of added training data for each iterative step
    nval: int
        The number of added validating data for each iterative step
    rmax: float
        The rmax input for NequIP
    lmax: int
        The lmax input for NequIP
    nfeatures: int
        The nfeatures input for NequIP
    workpath: str
        Path to the working directory
    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization
    """

    # Get a current path
    currentpath  = os.getcwd()

    if inputs.MLIP == 'nequip':
        job_script_input = []
        for index_nmodel in range(inputs.nmodel):
            for index_nstep in range(inputs.nstep):
                dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
                if os.path.exists(f'{workpath}/{dply_model}'):
                    single_print(f'\t\tFound the deployed model: {dply_model}')
                else: # If not, prepare the training job
                    rm_mkdir(f'{workpath}/train_{index_nmodel}_{index_nstep}')
                    job_script_input.append(nequip_train_job(
                        inputs, ntrain, nval, workpath, index_nstep, index_nmodel, dply_model
                    ))
                    single_print(f'\t\tPrepare a command line for traing a model: {dply_model}')

        for index_num_mdl in range(inputs.num_mdl_calc):

            # Prepare ingredients for the job script
            with open(f'./{inputs.job_MLIP_name}', 'r') as job_script_initial:
                job_script_default = job_script_initial.read()

            os.chdir(workpath)
            # Write an input for NequIP
            job_script   = f'./{inputs.job_MLIP_name.split(".")[0]}_{index_num_mdl}.{inputs.job_MLIP_name.split(".")[1]}'

            with open(job_script, 'w') as writing_input:
                writing_input.write(job_script_default)
                for index_item, job_item in enumerate(job_script_input):
                    if index_item % inputs.num_mdl_calc == index_num_mdl:
                        writing_input.write(job_item)

            # Auto-submission disabled by default. Uncomment to have
            # almomd dispatch each generated training job script directly.
            # subprocess.run([inputs.job_command, job_script]);
            os.chdir(currentpath)

    elif inputs.MLIP == 'so3krates':
        job_script_input = []
        for index_nmodel in range(inputs.nmodel):
            for index_nstep in range(inputs.nstep):
                dply_model = f'deployed-model_{index_nmodel}_{index_nstep}'
                if os.path.exists(f'{workpath}/{dply_model}'):
                    single_print(f'\t\tFound the deployed model: {dply_model}')
                else: # If not, prepare the training job
                    rm_mkdir(f'{workpath}/inputs_{index_nmodel}_{index_nstep}.json')
                    job_script_input.append(so3krates_train_job(
                        inputs, ntrain, nval, workpath, index_nstep, index_nmodel, dply_model
                    ))
                    single_print(f'\t\tPrepare a command line for traing a model: {dply_model}')

        for index_num_mdl in range(inputs.num_mdl_calc):
            inputs.job_MLIP_name = 'job-so3krates.slurm'
            # Prepare ingredients for the job script
            with open(f'./{inputs.job_MLIP_name}', 'r') as job_script_initial:
                job_script_default = job_script_initial.read()

            os.chdir(workpath)
            # Write an input for NequIP
            job_script   = f'./{inputs.job_MLIP_name.split(".")[0]}_{index_num_mdl}.{inputs.job_MLIP_name.split(".")[1]}'

            with open(job_script, 'w') as writing_input:
                writing_input.write(job_script_default)
                for index_item, job_item in enumerate(job_script_input):
                    if index_item % inputs.num_mdl_calc == index_num_mdl:
                        writing_input.write(job_item)

            # Submit the job scripts
            # subprocess.run([inputs.job_command, job_script]);
            os.chdir(currentpath)
