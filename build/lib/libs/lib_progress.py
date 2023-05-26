from ase.io.trajectory import Trajectory

import os
import numpy as np
import pandas as pd

from libs.lib_util        import check_mkdir, mpi_print, single_print
from libs.lib_struc       import check_struc
from libs.lib_criteria    import get_result
from libs.lib_train       import get_train_job
from libs.lib_npz         import generate_npz_DFT
from libs.lib_md          import check_runMD
from libs.lib_termination import termination


def check_progress(
    temperature, pressure, ensemble, timestep, friction,
    compressibility, taut, taup, mask, loginterval, steps_ther, name,
    supercell, ntotal, ntrain, ntrain_init, nval, nval_init, rmax, lmax,
    nfeatures, nstep, nmodel, steps_init, index, crtria, crtria_cnvg, NumAtoms,
    comm, size, rank
):
    kndex = 0
    MD_index = 0
    signal = 0
    
    outputfile = open('result.txt', 'w')
    outputfile.write(
        'Temperature[K]\tIteration\tUncerRel_E_init\tUncertAbs_E_init\tUncerRel_F_init\tUncertAbs_F_init\t'
        + 'UncertRel_E_All\tUncertAbs_E_All\tUncertRel_F_All\tUncertAbs_F_All\n'
    )
    outputfile.close()
    
    while True:
        uncert_file = f'uncertainty-{temperature}K-{pressure}bar_{index}.txt'

        if os.path.exists(f'./{uncert_file}'):
            uncert_data = pd.read_csv(uncert_file,\
                                      index_col=False, delimiter='\t')

            if len(uncert_data) == 0:
                trajfile = open(uncert_file, 'w')
                trajfile.write(
                    'Temperature[K]\tUncertRel_E\tUncertAbs_E\tUncertRel_F\tUncertAbs_F'
                    +'\tE_average\tCounting\tProbability\tAcceptance\n'
                )
                trajfile.close()
                break
            else:
                uncert_check = np.array(uncert_data.loc[:,'Counting'])[-1]
                del uncert_data

                if uncert_check[:7] == 'initial':
                    kndex = int(uncert_check[8:])
                    break
                elif uncert_check.replace(' ', '') == str(ntotal):
                    if rank == 0:
                        get_result(temperature, pressure, index, steps_init)
                    
                    aims_check = ['Have a nice day.' in open(f'calc/{temperature}K-{pressure}bar_{index+1}/{jndex}/aims/calculations/aims.out').read()\
                                   if os.path.exists(f'calc/{temperature}K-{pressure}bar_{index+1}/{jndex}/aims/calculations/aims.out') else False for jndex in range(nstep*(ntrain+nval))];
                    
                    if all(aims_check) == True:
                        index += 1
                        # Termination check
                        # signal = termination(temperature, pressure, crtria_cnvg, NumAtoms)
                        # if signal == 1:
                        #     mpi_print(f'{temperature}K is converged.', rank)
                    else:
                        single_print(f'DFT calculations of {temperature}K-{pressure}bar_{index+1} are not finished or not started.')
                        break
                else:
                    kndex = steps_init
                    MD_index = int(uncert_check)
                    break
        else:
            trajfile = open(uncert_file, 'w')
            trajfile.write(
                'Temperature[K]\tUncertRel_E\tUncertAbs_E\tUncertRel_F\tUncertAbs_F'
                +'\tE_average\tCounting\tProbability\tAcceptance\n'
            )
            trajfile.close()
            break
            
    return kndex, MD_index, index, signal

                          
def check_index(index):
    if os.path.exists('result.txt'):
        result_data = \
        pd.read_csv('result.txt', index_col=False, delimiter='\t')
        result_index = result_data.loc[:,'Iteration']; del result_data
        total_index = len(result_index)
    else:
        total_index = index
  
    return total_index


def check_result():
    logfile = 'result.txt'
    result_data = pd.read_csv(logfile, index_col=False, delim_whitespace=True)
    result_index = np.array(result_data.loc[:,'Iteration'])
    if len(result_index) > 0:
        index = result_index[-1]+1
    else:
        index = 0
        
    return index