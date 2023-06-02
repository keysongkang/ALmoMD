from ase.io.trajectory import Trajectory

import os
import numpy as np
import pandas as pd

from libs.lib_util        import check_mkdir, mpi_print, single_print
from libs.lib_criteria    import get_result, get_criteria
from libs.lib_train       import get_train_job
from libs.lib_termination import termination, get_testerror


def check_progress(
    temperature, pressure, ntotal, ntrain, nval,
    nstep, nmodel, steps_init, index, crtria, NumAtoms
):
    """Function [check_progress]
    Check the progress of previous calculations.
    Prepare the recording files if necessary.

    Parameters:

    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3

    ntotal: int
        Total number of added training and valdiation data for all subsamplings for each iteractive step
    ntrain: int
        The number of added training data for each iterative step
    nval: int
        The number of added validating data for each iterative step

    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization
    steps_init: int
        Initialize MD steps to get averaged uncertainties and energies
    index: int
        The index of AL interactive step

    crtria: float
        Convergence criteria
    NumAtoms: int
        The number of atoms in the simulation cell

    Returns:

    kndex: int
        The index for MLMD_init
    MD_index: int
        The index for MLMD_main
    index: int
        The index of AL interactive step
    signal: int
        The termination signal
    """

    from mpi4py import MPI

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialization
    kndex = 0
    MD_index = 0
    signal = 0
    
    if index == 0: # When calculation is just initiated
        # When there is no 'result.txt'
        if not os.path.exists('result.txt'):
            if rank == 0:
                # Open a recording 'result.txt' file
                outputfile = open('result.txt', 'w')
                outputfile.write(
                    'Temperature[K]\tIteration\t'
                    + 'TestError_E\tTestError_F\t'
                    + 'UncerRel_E_ini\tUncerAbs_E_ini\t'
                    + 'UncerRel_F_ini\tUncerAbs_F_ini\t'
                    + 'UncerRel_E_all\tUncerAbs_E_all\t'
                    + 'UncerRel_F_all\tUncerAbs_F_all\n'
                )
                outputfile.close()
            # Get the test errors using data-test.npz
            get_testerror(temperature, pressure, index, nstep, nmodel)
        else: # When there is a 'result.txt',
            # Check the contents in 'result.txt' before recording
            if os.path.exists('result.txt'):
                result_data = \
                pd.read_csv('result.txt', index_col=False, delimiter='\t')
                get_criteria_index = len(result_data.loc[:,'UncerAbs_F_all']);
            else:
                get_criteria_index = -1

            # Print the test errors only for first calculation
            if get_criteria_index == 0:
                # Get the test errors using data-test.npz
                get_testerror(temperature, pressure, index, nstep, nmodel)
    
    # Go through the while loop until a breaking command
    while True:
        # Uncertainty file
        uncert_file = f'UNCERT/uncertainty-{temperature}K-{pressure}bar_{index}.txt'

        # Check the existence of uncertainty file
        if os.path.exists(f'./{uncert_file}'):
            uncert_data = pd.read_csv(uncert_file,\
                                      index_col=False, delimiter='\t')

            if len(uncert_data) == 0: # If it is empty,
                if rank == 0:
                    check_mkdir('UNCERT')
                    trajfile = open(uncert_file, 'w')
                    trajfile.write(
                        'Temperature[K]\tUncertRel_E\tUncertAbs_E\tUncertRel_F\tUncertAbs_F'
                        +'\tEpot_average\tCounting\tProbability\tAcceptance\n'
                    )
                    trajfile.close()
                break
            else: # If it is not empty,
                # Check the last entry in the 'Couting' column
                uncert_check = np.array(uncert_data.loc[:,'Counting'])[-1]
                del uncert_data

                if uncert_check[:7] == 'initial': # If it has 'initial'
                    kndex = int(uncert_check[8:]) # Get the index of MLMD_init
                    break
                # If it reaches total number of sampling
                elif uncert_check.replace(' ', '') == str(ntotal):
                    # Record uncertainty results at the current step
                    junk = get_criteria(temperature, pressure, index, steps_init)
                    del junk
                    get_result(temperature, pressure, index, steps_init)
                    
                    # Check the FHI-vibes calculations
                    aims_check = ['Have a nice day.' in open(f'CALC/{temperature}K-{pressure}bar_{index+1}/{jndex}/aims/calculations/aims.out').read()\
                                   if os.path.exists(f'CALC/{temperature}K-{pressure}bar_{index+1}/{jndex}/aims/calculations/aims.out') else False for jndex in range(nstep*(ntrain+nval))];
                    
                    if all(aims_check) == True: # If all FHI-vibes calcs are finished,
                        gen_check = [
                        os.path.exists(f'MODEL/{temperature}K-{pressure}bar_{index+1}/deployed-model_{index_nmodel}_{index_nstep}.pth')
                        for index_nmodel in range(nmodel) for index_nstep in range(nstep)
                        ]

                        if all(gen_check) == True:
                            # Get the test errors using data-test.npz
                            index += 1
                            get_testerror(temperature, pressure, index, nstep, nmodel)
                            # Termination check
                            # signal = termination(temperature, pressure, crtria_cnvg, NumAtoms)
                            # if signal == 1:
                            #     mpi_print(f'{temperature}K is converged.', rank)
                        else:
                            index += 1
                            break

                    else: # Not finished
                        single_print(f'DFT calculations of {temperature}K-{pressure}bar_{index+1} are not finished or not started.')
                        signal = 1
                        break
                else: # Otherwise, get the index of MLMD_main
                    # Record uncertainty results at the current step
                    junk = get_criteria(temperature, pressure, index, steps_init)
                    del junk

                    kndex = steps_init
                    MD_index = int(uncert_check)
                    break
        else: # If there is no uncertainty file, create it
            if rank == 0:
                check_mkdir('UNCERT')
                trajfile = open(uncert_file, 'w')
                trajfile.write(
                    'Temperature[K]\tUncertRel_E\tUncertAbs_E\tUncertRel_F\tUncertAbs_F'
                    +'\tEpot_average\tCounting\tProbability\tAcceptance\n'
                )
                trajfile.close()
            break
            
    return kndex, MD_index, index, signal


def check_index(index):
    """Function [check_index]
    Check the progress of previous calculations
    and return the index of AL interactive steps.

    Parameters:

    index: int
        The index of AL interactive steps

    Returns:

    index: int
        The index of AL interactive steps from recorded file
    """

    # Open 'result.txt'
    if os.path.exists('result.txt'):
        result_data = \
        pd.read_csv('result.txt', index_col=False, delimiter='\t')
        result_index = len(result_data.loc[:,'Iteration']); del result_data
        if result_index > 0:
            total_index = result_index - 1
        else:
            total_index = result_index
    else:
        total_index = index
  
    return total_index
    