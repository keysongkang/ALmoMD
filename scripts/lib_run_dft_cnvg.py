import os
import sys
import numpy as np

from nequip.ase import nequip_calculator

from ase import Atoms

from libs.lib_util  import output_init, mpi_print, check_mkdir, eval_sigma

import torch
torch.set_default_dtype(torch.float64)


def run_dft_cnvg(inputs):
    """Function [run_dft_cnvg]
    Implement the convergence test with trained models
    """

    output_init('cnvg', inputs.version, inputs.rank)
    mpi_print(f'[cnvg]\tGet the convergence of {inputs.nmodel}x{inputs.nstep} matrix', inputs.rank)
    inputs.comm.Barrier()

    # Specify the path to the test data
    npz_test = f'./MODEL/data-test.npz'
    data_test = np.load(npz_test)
    mpi_print(f'[cnvg]\tLoad testing data: {npz_test}', inputs.rank)
    inputs.comm.Barrier()

    # Specify the working path and initialize the signal variable
    workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_0'
    signal = 0

    mpi_print(f'\t\tDevice: {inputs.device}', inputs.rank)

    mpi_print(f'[cnvg]\tFind the trained models: {workpath}', inputs.rank)
    # Load the trained models as calculators
    calc_MLIP = []
    for index_nmodel in range(inputs.nmodel):
        for index_nstep in range(inputs.nstep):
            if (index_nmodel * inputs.nstep + index_nstep) % inputs.size == inputs.rank:
                dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
                if os.path.exists(f'{workpath}/{dply_model}'):
                    mpi_print(f'\t\tFound the deployed model: {dply_model}', rank=0)
                    calc_MLIP.append(
                        nequip_calculator.NequIPCalculator.from_deployed_model(f'{workpath}/{dply_model}', device=inputs.device)
                    )
                else:
                    mpi_print(f'\t\tCannot find the model: {dply_model}', rank=0)
                    signal = 1
                    signal = inputs.comm.bcast(signal, root=inputs.rank)

    # Terminate the code when there is no tained model
    if signal == 1:
        mpi_print('[cnvg]\tNot enough trained models', inputs.rank)
        sys.exit()
    inputs.comm.Barrier()

    # Predict matrices of energy and forces and their R2 and MAE 
    mpi_print(f'[cnvg]\tGo through all trained models for the testing data', inputs.rank)

    prd_E_total = []
    prd_F_total = []
    prd_S_total = []

    for idx, (id_R, id_z, id_CELL, id_PBC) in enumerate(zip(
        data_test['R'], data_test['z'],
        data_test['CELL'], data_test['PBC']
    )):
        mpi_print(f'\t\t\tTesting data:{idx}', inputs.rank)

        struc = Atoms(
            id_z,
            positions=id_R,
            cell=id_CELL,
            pbc=id_PBC
        )

        prd_E = []
        prd_F = []
        prd_S = []
        zndex = 0
        for index_nmodel in range(inputs.nmodel):
            for index_nstep in range(inputs.nstep):
                if (index_nmodel * inputs.nstep + index_nstep) % inputs.size == inputs.rank:
                    struc.calc = calc_MLIP[zndex]
                    prd_E.append({f'{index_nmodel}_{index_nstep}': struc.get_potential_energy()})
                    prd_F.append({f'{index_nmodel}_{index_nstep}': struc.get_forces()})
                    prd_S.append({f'{index_nmodel}_{index_nstep}': eval_sigma(struc.get_forces(), struc.get_positions(), al_type='sigma')})
                    zndex += 1

        prd_E = inputs.comm.allgather(prd_E)
        prd_F = inputs.comm.allgather(prd_F)
        prd_S = inputs.comm.allgather(prd_S)

        prd_E_matrix = {}
        prd_F_matrix = {}
        prd_S_matrix = {}

        for item in prd_E:
            if item != []:
                for dict_item in item:
                    prd_E_matrix.update(dict_item)
        del prd_E
                        
        for item in prd_F:
            if item != []:
                for dict_item in item:
                    prd_F_matrix.update(dict_item)
        del prd_F

        for item in prd_S:
            if item != []:
                for dict_item in item:
                    prd_S_matrix.update(dict_item)
        del prd_S

        prd_E_total.append(prd_E_matrix)
        prd_F_total.append(prd_F_matrix)
        prd_S_total.append(prd_S_matrix)

        # if rank == 0:
        #     check_mkdir(f'E_matrix_prd')
        #     np.savez(f'E_matrix_prd/E_matrix_prd_{idx}', E = [prd_E_matrix])
        #     check_mkdir(f'F_matrix_prd')
        #     np.savez(f'F_matrix_prd/F_matrix_prd_{idx}', F = [prd_F_matrix])
        #     check_mkdir(f'S_matrix_prd')
        #     np.savez(f'S_matrix_prd/S_matrix_prd_{idx}', S = [prd_S_matrix])

    if inputs.rank == 0:
        np.savez(f'E_matrix_prd', E = prd_E_total)
        np.savez(f'F_matrix_prd', F = prd_F_total)
        np.savez(f'S_matrix_prd', S = prd_S_total)

    mpi_print(f'[cnvg]\tSave matrices: E_matrix, F_matrix, S_matrix', inputs.rank)