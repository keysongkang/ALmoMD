from nequip.ase import nequip_calculator

import os
import sys
import copy
import subprocess
import numpy as np

from libs.lib_util        import mpi_print, single_print, rm_mkdir


def get_train_job(
    struc_relaxed, ntrain, nval, rmax, lmax, nfeatures,
    workpath, nstep, nmodel, comm, size, rank
):
    signal    = 0
    currentpath  = os.getcwd()
    
    calc_MLIP = []
    for index_nmodel in range(nmodel):
        job_script_input = ''

        for index_nstep in range(nstep):
            dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
            if (index_nmodel*nstep + index_nstep) % size == rank:
                if os.path.exists(f'{workpath}/{dply_model}'):
                    mpi_print(f'Found the deployed model: {dply_model}', rank)
                    calc_MLIP.append(
                        nequip_calculator.NequIPCalculator.\
                        from_deployed_model(f'{workpath}/{dply_model}')
                    )
                else:
                    rm_mkdir(f'{workpath}/train_{index_nmodel}_{index_nstep}')
                    job_script_input += nequip_train_job(
                        ntrain, nval, rmax, lmax, nfeatures,
                        workpath, index_nstep, index_nmodel, dply_model, rank
                    )
                    mpi_print(f'Deploying the model: {dply_model}', rank)
                    signal = 1
                    signal = comm.bcast(signal, root=rank)

        job_script_input = comm.gather(job_script_input, root=0)
        if rank == 0:
            if job_script_input != ['' for i in range(size)]:
                # Prepare ingredients for the job script
                with open('./job-nequip-gpu.slurm', 'r') as job_script_initial:
                    job_script_default = job_script_initial.read()

                os.chdir(workpath)
                # Write an input for NequIP
                job_script   = f'./job-nequip-gpu_{index_nmodel}.slurm'

                with open(job_script, 'w') as writing_input:
                    writing_input.write(job_script_default)
                    for job_item in job_script_input:
                        writing_input.write(job_item)

                subprocess.run(['sbatch', job_script])
                os.chdir(currentpath)
                                
    if signal == 1:
        sys.exit()

    E_inter   = [] # Get ground state energies from trained models
    zndex = 0
    for index_nmodel in range(nmodel):
        for index_nstep in range(nstep):
            if (index_nmodel*nstep + index_nstep) % size == rank:
                struc_relaxed.calc = calc_MLIP[zndex]
                E_inter.append(struc_relaxed.get_potential_energy())
                zndex += 1
    E_inter = comm.allgather(E_inter)
    
    E_ref = np.average(np.array([i for items in E_inter for i in items]), axis=0);
    del E_inter;
    
    return E_ref, calc_MLIP


def nequip_train_job(
    ntrain, nval, rmax, lmax, nfeatures,
    workpath, index_nstep, index_nmodel, dply_model, rank
):
    deploy_model = f'./{dply_model}'
    
    # Check the current path
    currentpath  = os.getcwd()

    # Prepare ingredients for the NequIP input
    with open('./nequip.yaml', 'r') as nequip_initial:
        nequip_input_default = nequip_initial.read()

    nequip_input_extra = \
    f'root: train_{index_nmodel}_{index_nstep}/projects\n'\
    +f'run_name: train\n' + f'workdir: train\n'\
    +f'dataset_file_name: ./data-train_{index_nstep}.npz\n'\
    +f'n_train: {ntrain}\n' + f'n_val: {nval}\n'\
    +f'r_max: {rmax}\n' + f'l_max: {lmax}\n'\
    +f'num_features: {nfeatures}\n\n'\
    +f'seed: {index_nmodel}\n'
        
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

    os.chdir(currentpath)
    
    return f'{job_script_extra} {nequip_input}\n{job_script_deploy} {modelpath} {deploy_model}\n'
    
    
def execute_train_job(
    ntrain, nval, rmax, lmax, nfeatures,
    workpath, nstep, nmodel, comm, size, rank
):

    E_inter   = [] # Get ground state energies from trained models
    job_dependency = []
    currentpath  = os.getcwd()
    
    for index_nmodel in range(nmodel):
        job_script_input = ''

        for index_nstep in range(nstep):
            dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
            if (index_nmodel*nstep + index_nstep) % size == rank:
                if os.path.exists(f'{workpath}/{dply_model}'):
                    single_print(f'Found the deployed model: {dply_model}')
                else:
                    rm_mkdir(f'{workpath}/train_{index_nmodel}_{index_nstep}')
                    job_script_input += nequip_train_job(
                        ntrain, nval, rmax, lmax, nfeatures,
                        workpath, index_nstep, index_nmodel, dply_model, rank
                    )
                    single_print(f'Deploying the model: {dply_model}')

        job_script_input = comm.gather(job_script_input, root=0)
        if rank == 0:
            if job_script_input != ['' for i in range(size)]:
                # Prepare ingredients for the job script
                with open('./job-nequip-gpu.slurm', 'r') as job_script_initial:
                    job_script_default = job_script_initial.read()

                os.chdir(workpath)
                # Write an input for NequIP
                job_script   = f'./job-nequip-gpu_{index_nmodel}.slurm'

                with open(job_script, 'w') as writing_input:
                    writing_input.write(job_script_default)
                    for job_item in job_script_input:
                        writing_input.write(job_item)

                subprocess.run(['sbatch', job_script]);
                os.chdir(currentpath)