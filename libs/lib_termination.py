import os
import numpy as np
import pandas as pd
from libs.lib_util        import single_print


def termination(temperature, pressure, crtria_cnvg, al_type):
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
        if al_type == 'energy':
            result_uncert = result_data.loc[:,'TestError_E']
        elif al_type == 'force':
            result_uncert = result_data.loc[:,'TestError_F']
        elif al_type == 'force_max':
            result_uncert = result_data.loc[:,'UncerAbs_F_all']
        else:
            single_print('[term]\tYou need to assign al_teyp')

        # Verify three consecutive results against the convergence criteria
        if len(result_uncert) > 2:
            result_max = max(result_uncert[-3:])
            result_min = min(result_uncert[-3:])

            # Currently criteria is written by eV/atom
            if np.absolute(result_max - result_min) < crtria_cnvg and \
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


def get_testerror(temperature, pressure, index, nstep, nmodel):
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
    """
    from ase import Atoms
    from mpi4py import MPI
    from nequip.ase import nequip_calculator
    from libs.lib_util     import mpi_print
    from libs.lib_criteria import uncert_strconvter
    from sklearn.metrics import mean_absolute_error

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Read testing data
    npz_test = f'./MODEL/data-test.npz' # Path
    data_test = np.load(npz_test)         # Load
    NumAtom = len(data_test['z'][0])      # Get the number of atoms in the simulation cell

    # Set the path to folders storing the training data for NequIP
    workpath = f'./MODEL/{temperature}K-{pressure}bar_{index}'

    # Initialization of a termination signal
    signal = 0

    # Load the trained models as calculators
    calc_MLIP = []
    for index_nmodel in range(nmodel):
        for index_nstep in range(nstep):
            if (index_nmodel * nstep + index_nstep) % size == rank:
                dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
                if os.path.exists(f'{workpath}/{dply_model}'):
                    mpi_print(f'\t\tFound the deployed model: {dply_model}', rank=0)
                    calc_MLIP.append(
                        nequip_calculator.NequIPCalculator.from_deployed_model(f'{workpath}/{dply_model}')
                    )
                else:
                    # If there is no model, turn on the termination signal
                    mpi_print(f'\t\tCannot find the model: {dply_model}', rank=0)
                    signal = 1
                    signal = comm.bcast(signal, root=rank)

    # Check the termination signal
    if signal == 1:
        mpi_print('[Termi]\tSome training processes are not finished.', rank)
        sys.exit()

    # Go through all configurations in the testing data
    prd_E_list = []
    prd_F_list = []
    for id_R, id_z, id_CELL, id_PBC in zip(
        data_test['R'], data_test['z'], data_test['CELL'], data_test['PBC']
        ):
        # Create the corresponding ASE atoms
        struc = Atoms(id_z, positions=id_R, cell=id_CELL, pbc=id_PBC)
        # Prepare the empty lists for predicted energy and force
        prd_E = []
        prd_F = []
        zndex = 0

        # Go through all trained models
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                if (index_nmodel * nstep + index_nstep) % size == rank:
                    struc.calc = calc_MLIP[zndex]
                    # Predicted energy is shifted E_gs back, but the E_gs defualt is zero
                    prd_E.append(struc.get_potential_energy())
                    prd_F.append(struc.get_forces())
                    zndex += 1

        # Get average and standard deviation (Uncertainty) of predicted energies from various models
        prd_E = comm.allgather(prd_E)
        prd_E_list.append(np.average([jtem for item in prd_E if len(item) != 0 for jtem in item], axis=0))

        # Get average of predicted forces from various models
        prd_F = comm.allgather(prd_F)
        prd_F_list.append(np.average([jtem for item in prd_F if len(item) != 0 for jtem in item], axis=0))

    E_real = np.array(data_test['E']).flatten()
    E_pred = np.array(prd_E_list).flatten()
    F_real = np.array(data_test['F']).flatten()
    F_pred = np.array(prd_F_list).flatten()

    # Get the energy and force statistics
    E_MAE = mean_absolute_error(E_real, E_pred)
    F_MAE = mean_absolute_error(F_real, F_pred)

    if rank == 0:
        outputfile = open(f'result.txt', 'a')
        outputfile.write(
            f'{temperature}      \t{index}             \t' +
            uncert_strconvter(E_MAE) + '\t' +
            uncert_strconvter(F_MAE) + '\t'
            )
        outputfile.close()
