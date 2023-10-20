import os
import numpy as np
import pandas as pd
from libs.lib_util        import single_print

import torch
torch.set_default_dtype(torch.float64)

def termination(inputs):
    """Function [termination]
    Activate the termination signal upon satisfying the criteria

    Parameters:

    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    crtria: float
        Convergence criteria
    crtria_cnvg: float
        Convergence criteria ##!! We need to replace crtria_cnvg to crtria

    Returns:

    signal: int
        The termination signal
    """

    # Initialization
    signal = 0

    # Read the result file
    if os.path.exists('result.txt'):
        result_data = pd.read_csv(
            'result.txt', index_col=False, delimiter='\t'
            )
        # Read the Uncertainty column
        if inputs.al_type == 'energy':
            result_uncert = result_data.loc[:,'TestError_E']
        elif inputs.al_type == 'force':
            result_uncert = result_data.loc[:,'TestError_F']
        elif inputs.al_type == 'force_max':
            result_uncert = result_data.loc[:,'Un_Abs_F_std_i']
        elif inputs.al_type == 'sigma':
            result_uncert = result_data.loc[:,'TestError_S']
        elif inputs.al_type == 'sigma_max':
            result_uncert = result_data.loc[:,'Un_Abs_S_std_i']
        else:
            single_print('[term]\tYou need to assign al_teyp')

        # Verify three consecutive results against the convergence criteria
        if len(result_uncert) > 2:
            result_max = max(result_uncert[-3:])
            result_min = min(result_uncert[-3:])

            # Currently criteria is written by eV/atom
            if np.absolute(result_max - result_min) < inputs.crtria_cnvg and \
            result_max != result_min:
                single_print(
                    f'[term]\t!!The predicted results of trained model are'
                    +f' converged within the selected criteria:'
                    +f'{np.absolute(result_max-result_min)}'
                    )
                signal = 1
            else:
                signal = 0
        
    return signal


def get_testerror(inputs):
    """Function [get_testerror]
    Check the test error using data-test.npz.

    Parameters:

    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    index: int
        The index of AL interactive step
    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization
    calc_type: str
        Type of sampling; 'active' (active learning), 'random'
    """
    import sys
    from ase import Atoms
    from libs.lib_util     import mpi_print, eval_sigma
    from libs.lib_criteria import uncert_strconvter
    from sklearn.metrics import mean_absolute_error
    from ase.io import read as atoms_read

    # Read testing data
    npz_test = f'./MODEL/data-test.npz' # Path
    data_test = np.load(npz_test)         # Load

    ## Prepare the ground state structure
    # Read the ground state structure with the primitive cell
    struc_init = atoms_read('geometry.in.supercell', format='aims')
    # Get the number of atoms in the simulation cell

    # Get ground state energies predicted by trained models
    E_inter   = []
    zndex = 0
    for index_nmodel in range(inputs.nmodel):
        for index_nstep in range(inputs.nstep):
            if (index_nmodel*inputs.nstep + index_nstep) % inputs.size == inputs.rank:
                struc_init.calc = inputs.calc_MLIP[zndex]
                E_inter.append(struc_init.get_potential_energy())
                zndex += 1
    E_inter = inputs.comm.allgather(E_inter)
    
    # Calculate the average of ground state energies
    E_ref = np.average(np.array([i for items in E_inter for i in items]), axis=0);
    del E_inter;

    # Go through all configurations in the testing data
    prd_E_avg = []
    prd_E_std = []
    prd_F_avg = []
    prd_F_std = []
    prd_F_all_avg = []
    prd_sigma_avg = []
    prd_sigma_std = []
    prd_sigma_max_avg = []
    prd_sigma_max_std = []
    real_E_list = []
    real_F_list = []
    for id_step, (id_R, id_z, id_CELL, id_PBC, id_E, id_F) in enumerate(zip(
        data_test['R'], data_test['z'], data_test['CELL'], data_test['PBC'], data_test['E'], data_test['F']
        )):
        # Create the corresponding ASE atoms
        struc = Atoms(id_z, positions=id_R, cell=id_CELL, pbc=id_PBC)
        # Prepare the empty lists for predicted energy and force
        prd_E = []
        prd_F = []
        prd_R = []
        zndex = 0
        
        if id_step % inputs.printinterval == 0:
            mpi_print(f'[Termi]\tTesting: sample {id_step}', inputs.rank)

        # Go through all trained models
        for index_nmodel in range(inputs.nmodel):
            for index_nstep in range(inputs.nstep):
                if (index_nmodel * inputs.nstep + index_nstep) % inputs.size == inputs.rank:
                    struc.calc = inputs.calc_MLIP[zndex]
                    # Predicted energy is shifted E_gs back, but the E_gs defualt is zero
                    prd_E.append(struc.get_potential_energy())
                    prd_F.append(struc.get_forces(md=True))
                    prd_R.append(struc.get_positions())
                    zndex += 1

        # Get average and standard deviation (Uncertainty) of predicted energies from various models
        prd_E = inputs.comm.allgather(prd_E)
        prd_E = [jtem for item in prd_E if len(item) != 0 for jtem in item]
        inputs.comm.Barrier()

        # Get average of predicted forces from various models
        prd_F = inputs.comm.allgather(prd_F)
        prd_F = [jtem for item in prd_F if len(item) != 0 for jtem in item]

        if inputs.harmonic_F:
            from libs.lib_util import get_displacements, get_fc_ha, get_E_ha
            displacements = get_displacements(id_R, 'geometry.in.supercell')
            F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
            prd_F = prd_F + F_ha
            # id_F = id_F - F_ha
            E_ha = get_E_ha(displacements, F_ha)
            prd_E = np.array(prd_E) + E_ha
            # id_E = id_E - E_ha

        prd_E_avg.append(np.average(prd_E, axis=0))
        real_E_list.append(id_E)

        if inputs.al_type == 'energy':
            prd_E_std.append(np.std(prd_E, axis=0))

        prd_F_step_avg = np.average(prd_F, axis=0)
        prd_F_all_avg.append(prd_F_step_avg)
        real_F_list.append(id_F)

        if inputs.al_type == 'force' or inputs.al_type == 'force_max':
            prd_F_step_norm = np.array([[np.linalg.norm(Fcomp) for Fcomp in Ftems] for Ftems in prd_F - prd_F_step_avg])
            prd_F_step_norm_std = np.sqrt(np.average(prd_F_step_norm ** 2, axis=0))
            prd_F_step_norm_avg = np.linalg.norm(prd_F_step_avg, axis=1)

            prd_F_avg.append(prd_F_step_norm_avg)
            prd_F_std.append(prd_F_step_norm_std)

        # Get average of predicted sigmas from various models
        prd_R = inputs.comm.allgather(prd_R)
        prd_R = [jtem for item in prd_R if len(item) != 0 for jtem in item]

        prd_sigma = []
        prd_sigma_max = []
        for prd_F_step, prd_R_step in zip(prd_F, prd_R):
            prd_sigma.append(eval_sigma(prd_F_step, prd_R_step, al_type='sigma'))
            if inputs.al_type == 'sigma_max':
                prd_sigma_max.append(eval_sigma(prd_F_step, prd_R_step, inputs.al_type))

        prd_sigma_avg.append(np.average(prd_sigma, axis=0))
        if inputs.al_type == 'sigma':
            prd_sigma_std.append(np.std(prd_sigma, axis=0))
        elif inputs.al_type == 'sigma_max':
            prd_sigma_max_avg.append(np.average(prd_sigma_max, axis=0))
            prd_sigma_max_std.append(np.std(prd_sigma_max, axis=0))

    # E_real = np.array(data_test['E']).flatten()
    E_real = np.array(real_E_list).flatten()
    E_pred = np.array(prd_E_avg).flatten()
    F_real = np.array(real_F_list).flatten()
    F_pred = np.array(prd_F_all_avg).flatten()
    sigma_real = np.array(data_test['sigma']).flatten()
    sigma_pred = np.array(prd_sigma_avg).flatten()

    # Get the energy and force statistics
    E_MAE = mean_absolute_error(E_real, E_pred)
    F_MAE = mean_absolute_error(F_real, F_pred)
    sigma_MAE = mean_absolute_error(sigma_real, sigma_pred)

    prd_E_avg = np.array(prd_E_avg)-E_ref
    if inputs.al_type == 'energy':
        prd_E_std = np.array(prd_E_std)
        UncertAbs_E_avg = np.average(prd_E_std)
        UncertAbs_E_std = np.std(prd_E_std)
        UncertRel_E_avg = np.average(prd_E_std/prd_E_avg)
        UncertRel_E_std = np.std(prd_E_std/prd_E_avg)
    else:
        UncertAbs_E_avg = '----          '
        UncertAbs_E_std = '----          '
        UncertRel_E_avg = '----          '
        UncertRel_E_std = '----          '

    if inputs.al_type == 'force':
        prd_F_avg = np.array(prd_F_avg)
        prd_F_std = np.array(prd_F_std)
        UncertAbs_F_avg = np.average(prd_F_std)
        UncertAbs_F_std = np.average(prd_F_std)
        UncertRel_F_avg = np.average(prd_F_std/prd_F_avg)
        UncertRel_F_std = np.std(prd_F_std/prd_F_avg)
    elif inputs.al_type == 'force_max':
        UncertAbs_F_max = [max(prd_F_std_step) for prd_F_std_step in prd_F_std]
        UncertAbs_F_avg = np.average(UncertAbs_F_max)
        UncertAbs_F_std = np.std(UncertAbs_F_max)

        UncertRel_F_max = [max(prd_F_std_step/prd_F_avg_step) for prd_F_std_step, prd_F_avg_step in zip(prd_F_std, prd_F_avg)]
        UncertRel_F_avg = np.average(UncertRel_F_max)
        UncertRel_F_std = np.std(UncertRel_F_max)
    else:
        UncertAbs_F_avg = '----          '
        UncertAbs_F_std = '----          '
        UncertRel_F_avg = '----          '
        UncertRel_F_std = '----          '

    if inputs.al_type == 'sigma':
        prd_sigma_avg = np.array(prd_sigma_avg)
        prd_sigma_std = np.array(prd_sigma_std)
        UncertAbs_sigma_avg = np.average(prd_sigma_std)
        UncertAbs_sigma_std = np.std(prd_sigma_std)
        UncertRel_sigma_avg = np.average(prd_sigma_std/prd_sigma_avg)
        UncertRel_sigma_std = np.std(prd_sigma_std/prd_sigma_avg)
    elif inputs.al_type == 'sigma_max':
        UncertAbs_sigma_max = [max(prd_sigma_std_step) for prd_sigma_std_step in prd_sigma_max_std]
        UncertAbs_sigma_avg = np.average(UncertAbs_sigma_max)
        UncertAbs_sigma_std = np.std(UncertAbs_sigma_max)

        UncertRel_sigma_max = [max(prd_sigma_std_step/prd_sigma_avg_step) for prd_sigma_std_step, prd_sigma_avg_step in zip(prd_sigma_max_std, prd_sigma_max_avg)]
        UncertRel_sigma_avg = np.average(UncertRel_sigma_max)
        UncertRel_sigma_std = np.std(UncertRel_sigma_max)
    else:
        UncertAbs_sigma_avg = '----          '
        UncertAbs_sigma_std = '----          '
        UncertRel_sigma_avg = '----          '
        UncertRel_sigma_std = '----          '

    if inputs.calc_type == 'active' or inputs.calc_type == 'period':
        if inputs.rank == 0:
            outputfile = open(f'result.txt', 'a')

            result_print = f'{inputs.temperature}      \t{inputs.index}             '\
                           + '\t' + uncert_strconvter(E_MAE)\
                           + '\t' + uncert_strconvter(F_MAE)\
                           + '\t' + uncert_strconvter(sigma_MAE)\
                           + '\t' + uncert_strconvter(np.average(prd_E_avg))\
                           + '\t' + uncert_strconvter(np.std(prd_E_avg))

            if inputs.al_type == 'energy':
                result_print +=   '\t' + uncert_strconvter(UncertAbs_E_avg)\
                                + '\t' + uncert_strconvter(UncertAbs_E_std)\
                                + '\t' + uncert_strconvter(UncertRel_E_avg)\
                                + '\t' + uncert_strconvter(UncertRel_E_std)

            if inputs.al_type == 'force' or inputs.al_type == 'force_max':
                result_print +=   '\t' + uncert_strconvter(UncertAbs_F_avg)\
                                + '\t' + uncert_strconvter(UncertAbs_F_std)\
                                + '\t' + uncert_strconvter(UncertRel_F_avg)\
                                + '\t' + uncert_strconvter(UncertRel_F_std)

            if inputs.al_type == 'sigma' or inputs.al_type == 'sigma_max':
                result_print +=   '\t' + uncert_strconvter(UncertAbs_sigma_avg)\
                                + '\t' + uncert_strconvter(UncertAbs_sigma_std)\
                                + '\t' + uncert_strconvter(UncertRel_sigma_avg)\
                                + '\t' + uncert_strconvter(UncertRel_sigma_std)
            outputfile.write(result_print)
            outputfile.close()
    elif inputs.calc_type == 'random':
        if inputs.rank == 0:
            outputfile = open(f'result.txt', 'a')
            outputfile.write(
                f'{inputs.temperature}      \t{inputs.index}             \t' +
                uncert_strconvter(E_MAE) + '\t' +
                uncert_strconvter(F_MAE) + '\t' +
                uncert_strconvter(sigma_MAE) + '\n'
                )
            outputfile.close()
    else:
        mpi_print('You need to assign a clac_type', inputs.rank)