from ase.io.trajectory import Trajectory

import os
import numpy as np
import pandas as pd

from libs.lib_util        import check_mkdir, mpi_print, single_print
from libs.lib_criteria    import get_result
from libs.lib_train       import get_train_job
from libs.lib_md          import check_runMD
from libs.lib_termination import termination


def check_progress(
    temperature, pressure, ensemble, timestep, friction,
    compressibility, taut, taup, mask, loginterval, name,
    supercell, ntotal, ntrain, ntrain_init, nval, nval_init, rmax, lmax,
    nfeatures, nstep, nmodel, steps_init, index, crtria, crtria_cnvg, NumAtoms
):
    """Function [check_progress]
    Check the progress of previous calculations.
    Prepare the recording files if necessary.

    Parameters:

    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    ensemble: str
        Type of MD ensembles; 'NVTLangevin'  ##!! We don't need it anymore
    timestep: float
        The step interval for printing MD steps  ##!! We don't need it anymore
    friction: float
        Strength of the friction parameter in NVTLangevin ensemble  ##!! We don't need it anymore
    compressibility: float
        compressibility in units of eV/Angstrom**3 in NPTBerendsen  ##!! We don't need it anymore
    taut: float
        Time constant for Berendsen temperature coupling ##!! We don't need it anymore
        in NVTBerendsen and NPT Berendsen
    taup: float
        Time constant for Berendsen pressure coupling in NPTBerendsen ##!! We don't need it anymore
    mask: Three-element tuple
        Dynamic elements of the computational box (x,y,z); ##!! We don't need it anymore
        0 is false, 1 is true
    loginterval: int
        The step interval for printing MD steps ##!! We don't need it anymore
    name: str
        name of file ##!! We don't need it anymore
    supercell: list
        supersuper ##!! We don't need it anymore

    ntotal: int
        Total number of added training and valdiation data for all subsamplings for each iteractive step
    ntrain: int
        The number of added training data for each iterative step
    ntrain_init: int
        The number of training dat for initial step ##!! We don't need it anymore
    nval: int
        The number of added validating data for each iterative step
    nval_init:int
        The number of validating data for initial step ##!! We don't need it anymore

    rmax: float
        rmaxrmax ##!! We don't need it anymore
    lmax: int
        lmaxlmax ##!! We don't need it anymore
    nfeatures: int
        nfeatures ##!! We don't need it anymore

    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization ##!! We don't need it anymore
    steps_init:int
        Initialize MD steps to get averaged uncertainties and energies
    index: int
        The index of AL interactive step

    crtria: float
        Convergence criteria
    crtria_cnvg: float
        Convergence criteria ##!! We don't need it anymore

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

    # Initialization
    kndex = 0
    MD_index = 0
    signal = 0
    
    # Open a recording 'result.txt' file
    outputfile = open('result.txt', 'w')
    outputfile.write(
        'Temperature[K]\tIteration\tUncerRel_E_init\tUncertAbs_E_init\tUncerRel_F_init\tUncertAbs_F_init\t'
        + 'UncertRel_E_All\tUncertAbs_E_All\tUncertRel_F_All\tUncertAbs_F_All\n'
    )
    outputfile.close()
    
    # Go through the while loop until a breaking command
    while True:
        # Uncertainty file
        uncert_file = f'uncertainty-{temperature}K-{pressure}bar_{index}.txt'

        # Check the existence of uncertainty file
        if os.path.exists(f'./{uncert_file}'):
            uncert_data = pd.read_csv(uncert_file,\
                                      index_col=False, delimiter='\t')

            if len(uncert_data) == 0: # If it is empty,
                trajfile = open(uncert_file, 'w')
                trajfile.write(
                    'Temperature[K]\tUncertRel_E\tUncertAbs_E\tUncertRel_F\tUncertAbs_F'
                    +'\tE_average\tCounting\tProbability\tAcceptance\n'
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
                    get_result(temperature, pressure, index, steps_init)
                    
                    # Check the FHI-vibes calculations
                    aims_check = ['Have a nice day.' in open(f'calc/{temperature}K-{pressure}bar_{index+1}/{jndex}/aims/calculations/aims.out').read()\
                                   if os.path.exists(f'calc/{temperature}K-{pressure}bar_{index+1}/{jndex}/aims/calculations/aims.out') else False for jndex in range(nstep*(ntrain+nval))];
                    
                    if all(aims_check) == True: # If all FHI-vibes calcs are finished,
                        index += 1
                        # Termination check
                        # signal = termination(temperature, pressure, crtria_cnvg, NumAtoms)
                        # if signal == 1:
                        #     mpi_print(f'{temperature}K is converged.', rank)
                    else: # Not finished
                        single_print(f'DFT calculations of {temperature}K-{pressure}bar_{index+1} are not finished or not started.')
                        signal = 1
                        break
                else: # Otherwise, get the index of MLMD_main
                    kndex = steps_init
                    MD_index = int(uncert_check)
                    break
        else: # If there is no uncertainty file, create it
            trajfile = open(uncert_file, 'w')
            trajfile.write(
                'Temperature[K]\tUncertRel_E\tUncertAbs_E\tUncertRel_F\tUncertAbs_F'
                +'\tE_average\tCounting\tProbability\tAcceptance\n'
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
        result_index = result_data.loc[:,'Iteration']; del result_data
        total_index = len(result_index)
    else:
        total_index = index
  
    return total_index
    