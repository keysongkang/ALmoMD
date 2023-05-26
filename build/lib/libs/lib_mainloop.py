from ase.io.trajectory import Trajectory
from ase.io.trajectory import TrajectoryWriter

import os
import son
import copy
import random
import numpy as np
import pandas as pd
from ase import Atoms
from decimal import Decimal
from ase.data   import atomic_numbers

from libs.lib_util    import mpi_print, single_print, read_aims
from libs.lib_md       import runMD, check_runMD
from libs.lib_criteria import eval_uncert, uncert_strconvter, get_criteria, get_criteria_prob


def MLMD_initial(
    kndex, index, ensemble, temperature, pressure, timestep, friction,
    compressibility, taut, taup, mask, loginterval, steps_init, nstep,
    nmodel, calculator, E_ref, al_type, comm, size, rank
):

    if kndex == 0:
        if index == 0:
            metadata, traj = son.load('trajectory.son')
            traj_ther = traj[-1]
            struc_step = Atoms(
                [atomic_numbers[item[1]] for item in traj[-1]['atoms']['symbols'] for index in range(item[0])],
                positions = traj[-1]['atoms']['positions'],
                cell = traj[-1]['atoms']['cell'],
                pbc = traj[-1]['atoms']['pbc'],
                velocities = traj[-1]['atoms']['velocities']
            )
        else:
            uncert_file = f'uncertainty-{temperature}K-{pressure}bar_{index-1}.txt'
            if os.path.exists(uncert_file):
                #Start from a configuration with the largest real error
                struc_step = traj_fromRealE(temperature, pressure, E_ref, index)
                
                if rank == 0:
                    uncert_file_next = f'uncertainty-{temperature}K-{pressure}bar_{index}.txt'
                    trajfile = open(uncert_file_next, 'w')
                    trajfile.write(
                        'Temp.[K]\tUncertRel_E\tUncertAbs_E\tUncertRel_F\tUncertAbs_F\tE_average\tCounting\t'
                        + 'Probability\tAcceptance\n'
                    )
                    trajfile.close()
            else:
                traj_init     = f'traj-{temperature}K-{pressure}bar_{index}.traj'
                traj_previous = Trajectory(traj_init, properties=\
                                           ['forces', 'velocities', 'temperature'])
                struc_step    = traj_previous[-1]; del traj_previous;
    else:
        struc_step = []
        if rank == 0:
            traj_previous = Trajectory(f'temp-{temperature}K-{pressure}bar_{index}.traj',\
                                       properties=['forces', 'velocities', 'temperature'])
            struc_step    = traj_previous[-1]; del traj_previous;
        struc_step = comm.bcast(struc_step, root=0)
    
    workpath = f'./data/{temperature}K-{pressure}bar_{index}'
    for jndex in range(kndex, steps_init):
        # NO REAL ERROR ESTIMATION for MLMD1
        logfile      = f'temp-{temperature}K-{pressure}bar_{index}.log'
        trajectory   = f'temp-{temperature}K-{pressure}bar_{index}.traj'
        runMD(
            struc=struc_step, ensemble=ensemble, temperature=temperature,
            pressure=pressure, timestep=timestep, friction=friction,
            compressibility=compressibility, taut=taut, taup=taup,
            mask=mask, loginterval=loginterval, steps=loginterval,
            nstep=nstep, nmodel=nmodel, logfile=logfile,
            trajectory=trajectory, calculator=calculator,
            comm=comm, size=size, rank=rank
        )

        traj_current = Trajectory(trajectory, properties=\
                                  ['forces', 'velocities', 'temperature'])
        struc_step   = traj_current[-1]; del traj_current;

        UncertAbs_E, UncertRel_E, UncertAbs_F, UncertRel_F, Etot_step =\
        eval_uncert(struc_step, nstep, nmodel, E_ref, calculator, al_type, comm, size, rank)

        if rank == 0:
            trajfile = open(f'uncertainty-{temperature}K-{pressure}bar_{index}.txt', 'a')
            trajfile.write(
                '{:.5e}'.format(Decimal(str(struc_step.get_temperature()))) + '\t' +
                uncert_strconvter(UncertRel_E) + '\t' +
                uncert_strconvter(UncertAbs_E) + '\t' +
                uncert_strconvter(UncertRel_F) + '\t' +
                uncert_strconvter(UncertAbs_F) + '\t' +
                '{:.5e}'.format(Decimal(str(Etot_step))) + '\t' +
                f'initial_{jndex+1}' +
                '\t--         \t--         \t\n'
            )
            trajfile.close()
    
    mpi_print(
        f'The initial MLMD of the iteration {index}'
        + f'at {temperature}K and {pressure}bar is done', rank
    )
    
    return struc_step


def MLMD_main(
    MD_index, index, ensemble, temperature, pressure, timestep, friction,
    compressibility, taut, taup, mask, loginterval, ntotal, nstep, nmodel,
    calc_MLIP, E_ref, steps_init, NumAtoms, kB,
    struc_step, al_type, uncert_type, comm, size, rank
):

    # Extract the criteria from the initialization step
    criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std,\
    criteria_UncertRel_E_avg, criteria_UncertRel_E_std,\
    criteria_UncertAbs_F_avg, criteria_UncertAbs_F_std,\
    criteria_UncertRel_F_avg, criteria_UncertRel_F_std,\
    criteria_Etot_step_avg, criteria_Etot_step_std\
    = get_criteria(temperature, pressure, index, steps_init, size, rank)
    
    if index == 'init':
        write_traj =\
        TrajectoryWriter(filename=f'traj-{temperature}K-{pressure}bar_0.traj', mode='a')
    else:
        write_traj =\
        TrajectoryWriter(filename=f'traj-{temperature}K-{pressure}bar_{index+1}.traj', mode='a')

    if MD_index != 0:
        if rank == 0:
            traj_temp =\
            Trajectory(f'temp-{temperature}K-{pressure}bar_{index}.traj',\
                       properties='energy, forces')
            struc_step = traj_temp[-1]; del traj_temp;
        struc_step = comm.bcast(struc_step, root=0)
    
    workpath = f'./data/{temperature}K-{pressure}bar_{index}'
    
    while MD_index < ntotal:
        accept = '--         '

        logfile      = f'temp-{temperature}K-{pressure}bar_{index}.log'
        trajectory   = f'temp-{temperature}K-{pressure}bar_{index}.traj'
        
        runMD(
            struc_step, ensemble=ensemble, temperature=temperature,
            pressure=pressure, timestep=timestep, friction=friction,
            compressibility=compressibility, taut=taut, taup=taup,
            mask=mask, loginterval=loginterval, steps=loginterval,
            nstep=nstep, nmodel=nmodel, logfile=logfile,
            trajectory=trajectory, calculator=calc_MLIP,
            comm=comm, size=size, rank=rank
        )
        
        traj_current = Trajectory(trajectory,\
                                  properties=['forces', 'velocities', 'temperature'])
        struc_step   = traj_current[-1]; del traj_current;

        UncertAbs_E, UncertRel_E, UncertAbs_F, UncertRel_F, Etot_step =\
        eval_uncert(struc_step, nstep, nmodel, E_ref, calc_MLIP, al_type, comm, size, rank)
        
        criteria = get_criteria_prob(
            al_type, uncert_type, kB, NumAtoms, temperature, Etot_step, criteria_Etot_step_avg, criteria_Etot_step_std,
            UncertAbs_E, criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std, UncertRel_E, criteria_UncertRel_E_avg, criteria_UncertRel_E_std,
            UncertAbs_F, criteria_UncertAbs_F_avg, criteria_UncertAbs_F_std, UncertRel_F, criteria_UncertRel_F_avg, criteria_UncertRel_F_std
        )
        
        if rank == 0:
            if random.random() < criteria and Etot_step > 0.1: # Normalization
                accept = 'Accepted'
                MD_index += 1
                write_traj.write(atoms=struc_step)
            else:
                accept = 'Vetoed'

            trajfile = open(f'uncertainty-{temperature}K-{pressure}bar_{index}.txt', 'a')
            trajfile.write(
                '{:.5e}'.format(Decimal(str(struc_step.get_temperature()))) + '\t' +
                uncert_strconvter(UncertRel_E) + '\t' +
                uncert_strconvter(UncertAbs_E) + '\t' +
                uncert_strconvter(UncertRel_F) + '\t' +
                uncert_strconvter(UncertAbs_F) + '\t' +
                uncert_strconvter(Etot_step) + '\t' +
                str(MD_index) + '          \t' +
                '{:.5e}'.format(Decimal(str(criteria))) + '\t' +
                str(accept) + '   \n'
            )
            trajfile.close()
        MD_index = comm.bcast(MD_index, root=0)

    if rank == 0:
        single_print(f'The main MLMD of the iteration {index} at {temperature}K and {pressure}bar is done')

        
def traj_fromRealE(temperature, pressure, E_ref, index):
    uncertainty_data =\
    pd.read_csv(f'uncertainty-{temperature}K-{pressure}bar_{index-1}.txt',\
                index_col=False, delimiter='\t')

    RealError_data =\
    uncertainty_data.loc[uncertainty_data['Acceptance'] == 'Accepted   ']    
    
    max_RealError = 0
    max_index = 0
    for jndex in range(12):
        atoms, atoms_potE, atoms_forces = read_aims(f'calc/{temperature}K-{pressure}bar_1/{jndex}/aims/calculations/aims.out')
        RealError = np.absolute(np.array(RealError_data['E_average'])[jndex] + E_ref - atoms_potE)
        if RealError > max_RealError:
            max_RealError = RealError
            max_index = jndex
    
    traj = Trajectory(f'traj-{temperature}K-{pressure}bar_{index}.traj',\
                      properties=['forces', 'velocities', 'temperature'])
    struc = traj[max_index]
    
    return struc


def temp_initMD(
    ensemble, temperature, step_temp, pressure, timestep, friction,
    compressibility, taut, taup, mask, loginterval, steps_ther,
    nstep, nmodel, nototal_init, index, calc_DFT, comm, size, rank
):
    traj_lasttemp = f'traj-{temperature}K-{pressure}bar_{index-1}.traj'
    struc_lasttemp = Trajectory(traj_lasttemp, properties='energy, forces')
    
    temperature += step_temp
    index = 'init'
    signal = 0
    traj_index = check_runMD(
        struc_lasttemp[-1], ensemble, temperature, pressure, timestep,\
        friction, compressibility, taut, taup, mask, loginterval,\
        steps_ther, nstep, nmodel, index, calc_DFT, comm, size, rank
    ); del struc_lasttemp;

    index = 0
    struc_thermal = Trajectory(traj_index, properties='energy, forces')
    traj_index = check_runMD(
        struc_thermal[-1], ensemble, temperature, pressure, timestep,\
        friction, compressibility, taut, taup, mask, loginterval,\
        ntotal_init*loginterval, nstep, nmodel, index, calc_DFT,\
        comm, size, rank
    ); del struc_thermal;
    
    
def MLMD_random(
    kndex, index, ensemble, temperature, pressure, timestep, friction,
    compressibility, taut, taup, mask, loginterval, steps_init, nstep,
    nmodel, calculator, E_ref, comm, size, rank
):
    if kndex == 0:
        if index == 0:
            metadata, traj = son.load('trajectory.son')
            traj_ther = traj[-1]
            struc_step = Atoms(
                [atomic_numbers[item[1]] for item in traj[-1]['atoms']['symbols'] for index in range(item[0])],
                positions = traj[-1]['atoms']['positions'],
                cell = traj[-1]['atoms']['cell'],
                pbc = traj[-1]['atoms']['pbc'],
                velocities = traj[-1]['atoms']['velocities']
            )
        else:
            traj_init     = f'traj-{temperature}K-{pressure}bar_{index}.traj'
            traj_previous = Trajectory(traj_init, properties=\
                                       ['forces', 'velocities', 'temperature'])
            struc_step    = traj_previous[-1]; del traj_previous;
    else:
        struc_step = []
        if rank == 0:
            traj_previous = Trajectory(f'temp-{temperature}K-{pressure}bar_{index}.traj',\
                                       properties=['forces', 'velocities', 'temperature'])
            struc_step    = traj_previous[-1]; del traj_previous;
        struc_step = comm.bcast(struc_step, root=0)
    
    workpath = f'./data/{temperature}K-{pressure}bar_{index}'
    logfile      = f'traj-{temperature}K-{pressure}bar_{index+1}.log'
    trajectory   = f'traj-{temperature}K-{pressure}bar_{index+1}.traj'
    runMD(
        struc=struc_step, ensemble=ensemble, temperature=temperature,
        pressure=pressure, timestep=timestep, friction=friction,
        compressibility=compressibility, taut=taut, taup=taup,
        mask=mask, loginterval=loginterval, steps=steps_init,
        nstep=nstep, nmodel=nmodel, logfile=logfile,
        trajectory=trajectory, calculator=calculator,
        comm=comm, size=size, rank=rank
    )
    
    if rank == 0:
        criteriafile = open('result.txt', 'a')
        criteriafile.write(
            f'{temperature}          \t{index}          \n'
        )
        criteriafile.close()
    
    mpi_print(
        f'The initial MLMD of the iteration {index}'
        + f'at {temperature}K and {pressure}bar is done', rank
    )
    
    return struc_step