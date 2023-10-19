from ase.io.trajectory import Trajectory
from ase.io.trajectory import TrajectoryWriter

import os
import son
import copy
import random
import numpy as np
import pandas as pd
from ase import Atoms
from mpi4py import MPI
from decimal import Decimal
from ase.build import make_supercell
from ase.io import read as atoms_read
from ase.data   import atomic_numbers
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from libs.lib_util    import check_mkdir, mpi_print, single_print, read_aims
from libs.lib_md       import runMD
from libs.lib_criteria import eval_uncert, uncert_strconvter, get_criteria, get_criteria_prob



def MLMD_initial(
    kndex, index, ensemble, temperature, pressure, timestep, friction,
    compressibility, taut, taup, mask, loginterval, steps_init, nstep,
    nmodel, calculator, E_ref, al_type, harmonic_F, anharmonic_F
):
    """Function [MLMD_initial]
    Initiate the Molecular Dynamics with trained model
    to get the average and standard deviation
    of uncertainty and total energies

    Parameters:

    kndex: int
        The index for MLMD_initial steps
        It is used to resume a terminated calculation
    index: int
        The index of AL interactive step
    ensemble: str
        Type of MD ensembles; 'NVTLangevin'
    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    timestep: float
        The step interval for printing MD steps

    friction: float
        Strength of the friction parameter in NVTLangevin ensemble
    compressibility: float
        compressibility in units of eV/Angstrom**3 in NPTBerendsen
    taut: float
        Time constant for Berendsen temperature coupling
        in NVTBerendsen and NPT Berendsen
    taup: float
        Time constant for Berendsen pressure coupling in NPTBerendsen
    mask: Three-element tuple
        Dynamic elements of the computational box (x,y,z);
        0 is false, 1 is true

    loginterval: int
        The step interval for printing MD steps
    steps_init: int
        The number of initialization MD steps
        to get averaged uncertainties and energies

    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization
    calculator: ASE calculator
        Calculators from trained models
    E_ref: flaot
        The energy of reference state (Here, ground state)
    al_type: str
        Type of active learning: 'energy', 'force', 'force_max'

    Returns:

    struc_step: ASE atoms
        Last entry of the configuration from initial trajectory
    """

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # When this initialization starts from scratch,
    if kndex == 0:
        # Even when it is very first iterative step,
        if index == 0:
            if os.path.exists('start.in'):
                # Read the ground state structure with the primitive cell
                struc_init = atoms_read(self.MD_input, format='aims')
                # Make it supercell
                struc_step = make_supercell(struc_init, self.supercell_init)
                MaxwellBoltzmannDistribution(struc, temperature_K=self.temperature, force_temp=True)
            else:
                # Read the trajectory file from 'trajectory_train.son'
                metadata, traj = son.load('trajectory_train.son')
                # Take the last configuration from 'trajectory_train.son'
                traj_ther = traj[-1]
                # Convert 'trajectory.son' format to ASE atoms
                struc_step = Atoms(
                    [atomic_numbers[item[1]] for item in traj[-1]['atoms']['symbols'] for index in range(item[0])],
                    positions = traj[-1]['atoms']['positions'],
                    cell = traj[-1]['atoms']['cell'],
                    pbc = traj[-1]['atoms']['pbc'],
                    velocities = traj[-1]['atoms']['velocities']
                )
        else: # When it has the pervious step,
            # # Name of the pervious uncertainty file
            # uncert_file = f'UNCERT/uncertainty-{temperature}K-{pressure}bar_{index-1}.txt'

            # # Check the existence of the file
            # if os.path.exists(uncert_file):
            #     # Start from the configuration with the largest real error
            #     struc_step = traj_fromRealE(temperature, pressure, E_ref, index)
                
            # else: # If there is no privous uncertainty file
            # Read the trajectory from previous trajectory file
            traj_init     = f'TRAJ/traj-{temperature}K-{pressure}bar_{index}.traj'
            traj_previous = Trajectory(traj_init, properties=\
                                       ['forces', 'velocities', 'temperature'])
            # Resume the MD calculation from last configuration in the trajectory file
            struc_step    = traj_previous[-1]; del traj_previous;

            # Open the uncertainty file for current step
            if rank == 0:
                check_mkdir('UNCERT')
                uncert_file_next = f'UNCERT/uncertainty-{temperature}K-{pressure}bar_{index}.txt'
                trajfile = open(uncert_file_next, 'w')
                trajfile.write(
                    'Temperature[K]\tUncertAbs_E\tUncertRel_E\tUncertAbs_F\tUncertRel_F'
                    +'\tUncertAbs_S\tUncertRel_S\tEpot_average\tS_average'
                    +'\tCounting\tProbability\tAcceptance\n'
                )
                trajfile.close()

    else: # If it starts from terminated point,
        struc_step = []
        if rank == 0:
            # Read the trajectory from previous file
            traj_previous = Trajectory(
                f'TEMPORARY/temp-{temperature}K-{pressure}bar_{index}.traj',
                properties=['forces', 'velocities', 'temperature']
                )
            struc_step = traj_previous[-1]; del traj_previous;
        # Resume the MD calculatio nfrom last configuration
        struc_step = comm.bcast(struc_step, root=0)
    
    # Initiate the MD run starting from kndex until reaching steps_init
    for jndex in range(kndex, steps_init):
        # MD information for temporary steps
        check_mkdir('TEMPORARY')
        trajectory   = f'TEMPORARY/temp-{temperature}K-{pressure}bar_{index}.traj'

        # Implement MD calculation for only one loginterval step
        runMD(
            struc=struc_step, ensemble=ensemble, temperature=temperature,
            pressure=pressure, timestep=timestep, friction=friction,
            compressibility=compressibility, taut=taut, taup=taup,
            mask=mask, loginterval=loginterval, steps=loginterval,
            nstep=nstep, nmodel=nmodel, E_ref=E_ref, al_type=al_type,
            logfile=None, trajectory=trajectory, calculator=calculator,
            harmonic_F=harmonic_F, anharmonic_F=anharmonic_F, signal_uncert=False, signal_append=False
        )

        # Get new configuration and velocities for next step
        traj_current = Trajectory(
            trajectory, properties=['forces', 'velocities', 'temperature']
            )
        struc_step   = traj_current[-1]
        del traj_current # Remove it to reduce the memory usage

        # Get absolute and relative uncertainties of energy and force
        # and also total energy
        UncertAbs_E, UncertRel_E, UncertAbs_F, UncertRel_F, Epot_step =\
        eval_uncert(struc_step, nstep, nmodel, E_ref, calculator, al_type, harmonic_F)

        # Record the all uncertainty and total energy information at the current step
        if rank == 0:
            trajfile = open(f'UNCERT/uncertainty-{temperature}K-{pressure}bar_{index}.txt', 'a')
            trajfile.write(
                '{:.5e}'.format(Decimal(str(struc_step.get_temperature()))) + '\t' +
                uncert_strconvter(UncertAbs_E) + '\t' +
                uncert_strconvter(UncertRel_E) + '\t' +
                uncert_strconvter(UncertAbs_F) + '\t' +
                uncert_strconvter(UncertRel_F) + '\t' +
                '{:.5e}'.format(Decimal(str(Epot_step))) + '\t' +
                f'initial_{jndex+1}' +
                '\t--         \t--         \t\n'
            )
            trajfile.close()
    
    mpi_print(
        f'[MLMD_init]\tThe initial MLMD of the iteration {index}'
        + f'at {temperature}K and {pressure}bar is done', rank
    )
    
    return struc_step



def MLMD_main(
    inputs, MD_index, calc_MLIP, E_ref
):
    """Function [MLMD_main]
    Initiate the Molecular Dynamics with trained model
    to sample the configuration satisfying the active learning critria

    Parameters:

    MD_index: int
        The index for MLMD_main steps
        It also indicates the number of sampled configurations
        It is used to resume a terminated calculation
    index: int
        The index of AL interactive step
    ensemble: str
        Type of MD ensembles; 'NVTLangevin'
    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    timestep: float
        The step interval for printing MD steps

    friction: float
        Strength of the friction parameter in NVTLangevin ensemble
    compressibility: float
        compressibility in units of eV/Angstrom**3 in NPTBerendsen
    taut: float
        Time constant for Berendsen temperature coupling
        in NVTBerendsen and NPT Berendsen
    taup: float
        Time constant for Berendsen pressure coupling in NPTBerendsen
    mask: Three-element tuple
        Dynamic elements of the computational box (x,y,z);
        0 is false, 1 is true

    loginterval: int
        The step interval for printing MD steps

    ntotal: int
        The total number of sampling data
    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization
    calc_MLIP: ASE calculator
        Calculators from trained models
    E_ref: flaot
        The energy of reference state (Here, ground state)
    steps_init: int
        The number of initialization MD steps
        to get averaged uncertainties and energies
    al_type: str
        Type of active learning: 'energy', 'force', 'force_max'
    uncert_type: str
        Type of uncertainty; 'absolute', 'relative'
    uncert_shift: float
        Shifting of erf function
        (Value is relative to standard deviation)
    uncert_grad: float
        Gradient of erf function
        (Value is relative to standard deviation)
    """

    # Extract the criteria information from the initialization step
    criteria_Epot_step_avg,   criteria_Epot_step_std,\
    criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std,\
    criteria_UncertRel_E_avg, criteria_UncertRel_E_std,\
    criteria_UncertAbs_F_avg, criteria_UncertAbs_F_std,\
    criteria_UncertRel_F_avg, criteria_UncertRel_F_std,\
    criteria_UncertAbs_S_avg, criteria_UncertAbs_S_std,\
    criteria_UncertRel_S_avg, criteria_UncertRel_S_std\
    = get_criteria(inputs.temperature, inputs.pressure, inputs.index, inputs.steps_init, inputs.al_type)
    
    # Open a trajectory file to store the sampled configurations
    if inputs.rank == 0:
        check_mkdir('TEMPORARY')
        check_mkdir('TRAJ')
    write_traj = TrajectoryWriter(
        filename=f'TRAJ/traj-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index+1}.traj',
        mode='a'
        )

    mpi_print(f'[MLMD] Calculate it from index {inputs.index} and MD_index {MD_index}', inputs.rank)
    # When this initialization starts from scratch,
    if MD_index == 0:
        # Even when it is very first iterative step,
        if inputs.index != 0:
            # Name of the pervious uncertainty file
            uncert_file = f'UNCERT/uncertainty-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index-1}.txt'

            # Check the existence of the file
            if os.path.exists(uncert_file):
                # Start from the configuration with the largest real error
                struc_step = traj_fromRealE(inputs.temperature, inputs.pressure, E_ref, inputs.E_gs, inputs.uncert_type, inputs.al_type, inputs.ntotal, inputs.index)
                
                # Open the uncertainty file for current step
                if inputs.rank == 0:
                    check_mkdir('UNCERT')
                    uncert_file_next = f'UNCERT/uncertainty-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}.txt'
                    trajfile = open(uncert_file_next, 'w')
                    trajfile.write(
                        'Temperature[K]\tUncertAbs_E\tUncertRel_E\tUncertAbs_F\tUncertRel_F'
                        +'\tUncertAbs_S\tUncertRel_S\tEpot_average\tS_average'
                        +'\tCounting\tProbability\tAcceptance\n'
                    )
                    trajfile.close()

            else: # If there is no privous uncertainty file
                traj_temp     = f'TEMPORARY/temp-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}.traj'
                mpi_print(f'[MLMD] Read a configuration from traj file', inputs.rank)
                # Read the trajectory from previous trajectory file
                traj_previous = Trajectory(traj_temp, properties=\
                                           ['forces', 'velocities', 'temperature'])
                # Resume the MD calculation from last configuration in the trajectory file
                struc_step    = traj_previous[-1]; del traj_previous;

        elif os.path.exists('start.in'):
            mpi_print(f'[MLMD] Read a configuration from start.in', inputs.rank)
            # Read the ground state structure with the primitive cell
            struc_init = atoms_read(inputs.MD_input, format='aims')
            # Make it supercell
            struc_step = make_supercell(struc_init, inputs.supercell_init)
            MaxwellBoltzmannDistribution(struc, temperature_K=inputs.temperature, force_temp=True)

        else:
            mpi_print(f'[MLMD] Read a configuration from trajectory_train.son', inputs.rank)
            # Read the trajectory file from 'trajectory_train.son'
            metadata, traj = son.load('trajectory_train.son')
            # Take the last configuration from 'trajectory_train.son'
            traj_ther = traj[-1]
            # Convert 'trajectory.son' format to ASE atoms
            struc_step = Atoms(
                [atomic_numbers[item[1]] for item in traj[-1]['atoms']['symbols'] for xdx in range(item[0])],
                positions = traj[-1]['atoms']['positions'],
                cell = traj[-1]['atoms']['cell'],
                pbc = traj[-1]['atoms']['pbc'],
                velocities = traj[-1]['atoms']['velocities']
            )
    else: # If it starts from terminated point,
        mpi_print(f'[MLMD] Read a configuration from temp file', inputs.rank)
        if inputs.rank == 0:
            # Read the trajectory from previous file
            traj_previous = Trajectory(
                f'TEMPORARY/temp-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}.traj',
                properties=['forces', 'velocities', 'temperature']
                )
            struc_step = traj_previous[-1]; del traj_previous;
        # Resume the MD calculatio nfrom last configuration
        struc_step = inputs.comm.bcast(struc_step, root=0)

    # Initiate the MD run starting from MD_index until reaching ntotal
    # MD_index also indicates the number of sampled configurations
    while MD_index < inputs.ntotal:
        accept = '--         '

        # MD information for temporary steps
        trajectory   = f'TEMPORARY/temp-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}.traj'
        
        mpi_print(f'[MLMD] Run MD: iteration {MD_index}', inputs.rank)
        # Implement the MD calculation for only one loginterval step
        runMD(
            inputs, struc_step, inputs.loginterval,
            logfile=None, trajectory=trajectory, calculator=calc_MLIP,
            signal_uncert=False, signal_append=False
        )
        
        # Get new configuration and velocities for next step
        traj_current = Trajectory(
            trajectory,
            properties=['forces', 'velocities', 'temperature']
            )
        struc_step   = traj_current[-1]
        del traj_current # Remove it to reduce the memory usage

        # Get absolute and relative uncertainties of energy and force
        # and also total energy
        UncertAbs_E, UncertRel_E, UncertAbs_F, UncertRel_F, UncertAbs_S, UncertRel_S, Epot_step, S_step =\
        eval_uncert(struc_step, inputs.nstep, inputs.nmodel, 0.0, calc_MLIP, inputs.al_type, inputs.harmonic_F)
        
        # Get a criteria probability from uncertainty and energy informations
        criteria = get_criteria_prob(
            inputs.al_type, inputs.uncert_type, inputs.uncert_shift, inputs.uncert_grad,
            inputs.kB, inputs.NumAtoms, inputs.temperature,
            Epot_step,   criteria_Epot_step_avg,   criteria_Epot_step_std,
            UncertAbs_E, criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std,
            UncertRel_E, criteria_UncertRel_E_avg, criteria_UncertRel_E_std,
            UncertAbs_F, criteria_UncertAbs_F_avg, criteria_UncertAbs_F_std,
            UncertRel_F, criteria_UncertRel_F_avg, criteria_UncertRel_F_std,
            UncertAbs_S, criteria_UncertAbs_S_avg, criteria_UncertAbs_S_std,
            UncertRel_S, criteria_UncertRel_S_avg, criteria_UncertRel_S_std
        )

        if inputs.rank == 0:
            # Acceptance check with criteria
            ##!! Epot_step should be rechecked.
            if random.random() < criteria and Epot_step > 0.1:
                accept = 'Accepted'
                MD_index += 1
                write_traj.write(atoms=struc_step)
            else:
                accept = 'Vetoed'

            # Record the MD results at the current step
            trajfile = open(f'UNCERT/uncertainty-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}.txt', 'a')
            trajfile.write(
                '{:.5e}'.format(Decimal(str(struc_step.get_temperature()))) + '\t' +
                uncert_strconvter(UncertAbs_E) + '\t' +
                uncert_strconvter(UncertRel_E) + '\t' +
                uncert_strconvter(UncertAbs_F) + '\t' +
                uncert_strconvter(UncertRel_F) + '\t' +
                uncert_strconvter(UncertAbs_S) + '\t' +
                uncert_strconvter(UncertRel_S) + '\t' +
                uncert_strconvter(Epot_step) + '\t' +
                uncert_strconvter(S_step) + '\t' +
                str(MD_index) + '          \t' +
                '{:.5e}'.format(Decimal(str(criteria))) + '\t' +
                str(accept) + '   \n'
            )
            trajfile.close()
        MD_index = inputs.comm.bcast(MD_index, root=0)

    mpi_print(f'[MLMD_main]\tThe main MLMD of the iteration {inputs.index} at {inputs.temperature}K and {inputs.pressure}bar is done', inputs.rank)


def MLMD_main_period(
    inputs, MD_index, MD_step_index, calc_MLIP, E_ref
):
    """Function [MLMD_main]
    Initiate the Molecular Dynamics with trained model
    to sample the configuration satisfying the active learning critria

    Parameters:

    MD_index: int
        The index for MLMD_main steps
        It also indicates the number of sampled configurations
        It is used to resume a terminated calculation
    index: int
        The index of AL interactive step
    ensemble: str
        Type of MD ensembles; 'NVTLangevin'
    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    timestep: float
        The step interval for printing MD steps

    friction: float
        Strength of the friction parameter in NVTLangevin ensemble
    compressibility: float
        compressibility in units of eV/Angstrom**3 in NPTBerendsen
    taut: float
        Time constant for Berendsen temperature coupling
        in NVTBerendsen and NPT Berendsen
    taup: float
        Time constant for Berendsen pressure coupling in NPTBerendsen
    mask: Three-element tuple
        Dynamic elements of the computational box (x,y,z);
        0 is false, 1 is true

    loginterval: int
        The step interval for printing MD steps

    ntotal: int
        The total number of sampling data
    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization
    calc_MLIP: ASE calculator
        Calculators from trained models
    E_ref: flaot
        The energy of reference state (Here, ground state)
    steps_init: int
        The number of initialization MD steps
        to get averaged uncertainties and energies
    al_type: str
        Type of active learning: 'energy', 'force', 'force_max'
    uncert_type: str
        Type of uncertainty; 'absolute', 'relative'
    uncert_shift: float
        Shifting of erf function
        (Value is relative to standard deviation)
    uncert_grad: float
        Gradient of erf function
        (Value is relative to standard deviation)
    """

    # Extract the criteria information from the initialization step
    criteria_Epot_step_avg,   criteria_Epot_step_std,\
    criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std,\
    criteria_UncertRel_E_avg, criteria_UncertRel_E_std,\
    criteria_UncertAbs_F_avg, criteria_UncertAbs_F_std,\
    criteria_UncertRel_F_avg, criteria_UncertRel_F_std,\
    criteria_UncertAbs_S_avg, criteria_UncertAbs_S_std,\
    criteria_UncertRel_S_avg, criteria_UncertRel_S_std\
    = get_criteria(inputs.temperature, inputs.pressure, inputs.index, inputs.steps_init, inputs.al_type)
    
    # Open a trajectory file to store the sampled configurations
    if inputs.rank == 0:
        check_mkdir('TEMPORARY')
        check_mkdir('TRAJ')
    write_traj = TrajectoryWriter(
        filename=f'TRAJ/traj-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index+1}.traj',
        mode='a'
        )

    mpi_print(f'[MLMD] Calculate it from index {inputs.index}, MD_index {MD_index}, MD_step_index {MD_step_index}', inputs.rank)
    # When this initialization starts from scratch,
    if MD_index == 0:
        # Even when it is very first iterative step,
        if inputs.index != 0:
            # Name of the pervious uncertainty file
            uncert_file = f'UNCERT/uncertainty-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index-1}.txt'

            # Check the existence of the file
            if os.path.exists(uncert_file):
                # Start from the configuration with the largest real error
                struc_step = traj_fromRealE(inputs.temperature, inputs.pressure, 0.0, inputs.E_gs, inputs.uncert_type, inputs.al_type, inputs.ntotal, inputs.index)
                
                # Open the uncertainty file for current step
                if inputs.rank == 0:
                    check_mkdir('UNCERT')
                    uncert_file_next = f'UNCERT/uncertainty-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}.txt'
                    trajfile = open(uncert_file_next, 'w')
                    trajfile.write(
                        'Temperature[K]\tUncertAbs_E\tUncertRel_E\tUncertAbs_F\tUncertRel_F'
                        +'\tUncertAbs_S\tUncertRel_S\tEpot_average\tS_average'
                        +'\tCounting\tProbability\tAcceptance\n'
                    )
                    trajfile.close()

            else: # If there is no privous uncertainty file
                traj_temp     = f'TEMPORARY/temp-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}.traj'
                mpi_print(f'[MLMD] Read a configuration from traj file', inputs.rank)
                # Read the trajectory from previous trajectory file
                traj_previous = Trajectory(traj_temp, properties=\
                                           ['forces', 'velocities', 'temperature'])
                # Resume the MD calculation from last configuration in the trajectory file
                struc_step    = traj_previous[-1]; del traj_previous;

        elif os.path.exists('start.in'):
            mpi_print(f'[MLMD] Read a configuration from start.in', inputs.rank)
            # Read the ground state structure with the primitive cell
            struc_init = atoms_read(inputs.MD_input, format='aims')
            # Make it supercell
            struc_step = make_supercell(struc_init, inputs.supercell_init)
            MaxwellBoltzmannDistribution(struc, temperature_K=inputs.temperature, force_temp=True)

        else:
            mpi_print(f'[MLMD] Read a configuration from trajectory_train.son', inputs.rank)
            # Read the trajectory file from 'trajectory_train.son'
            metadata, traj = son.load('trajectory_train.son')
            # Take the last configuration from 'trajectory_train.son'
            traj_ther = traj[-1]
            # Convert 'trajectory.son' format to ASE atoms
            struc_step = Atoms(
                [atomic_numbers[item[1]] for item in traj[-1]['atoms']['symbols'] for inputs.index in range(item[0])],
                positions = traj[-1]['atoms']['positions'],
                cell = traj[-1]['atoms']['cell'],
                pbc = traj[-1]['atoms']['pbc'],
                velocities = traj[-1]['atoms']['velocities']
            )
    else: # If it starts from terminated point,
        mpi_print(f'[MLMD] Read a configuration from temp file', inputs.rank)
        if inputs.rank == 0:
            # Read the trajectory from previous file
            traj_previous = Trajectory(
                f'TEMPORARY/temp-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}.traj',
                properties=['forces', 'velocities', 'temperature']
                )
            struc_step = traj_previous[-1]; del traj_previous;
        # Resume the MD calculatio nfrom last configuration
        struc_step = inputs.comm.bcast(struc_step, root=0)

    # Initiate the MD run starting from MD_index until reaching ntotal
    # MD_index also indicates the number of sampled configurations
    while MD_index < inputs.ntotal or MD_step_index < inputs.nperiod:
        accept = '--         '

        # MD information for temporary steps
        trajectory   = f'TEMPORARY/temp-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}.traj'
        
        mpi_print(f'[MLMD] Run MD: iteration {MD_index}', inputs.rank)
        # Implement the MD calculation for only one loginterval step
        runMD(
            inputs, struc_step, inputs.loginterval,
            logfile=None, trajectory=trajectory, calculator=calc_MLIP,
            signal_uncert=False, signal_append=False
        )
        
        # Get new configuration and velocities for next step
        traj_current = Trajectory(
            trajectory,
            properties=['forces', 'velocities', 'temperature']
            )
        struc_step   = traj_current[-1]
        del traj_current # Remove it to reduce the memory usage

        # Get absolute and relative uncertainties of energy and force
        # and also total energy
        UncertAbs_E, UncertRel_E, UncertAbs_F, UncertRel_F, UncertAbs_S, UncertRel_S, Epot_step, S_step =\
        eval_uncert(struc_step, inputs.nstep, inputs.nmodel, 0.0, calc_MLIP, inputs.al_type, inputs.harmonic_F)
        
        # Get a criteria probability from uncertainty and energy informations
        criteria = get_criteria_prob(
            inputs.al_type, inputs.uncert_type, inputs.uncert_shift, inputs.uncert_grad,
            inputs.kB, inputs.NumAtoms, inputs.temperature,
            Epot_step,   criteria_Epot_step_avg,   criteria_Epot_step_std,
            UncertAbs_E, criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std,
            UncertRel_E, criteria_UncertRel_E_avg, criteria_UncertRel_E_std,
            UncertAbs_F, criteria_UncertAbs_F_avg, criteria_UncertAbs_F_std,
            UncertRel_F, criteria_UncertRel_F_avg, criteria_UncertRel_F_std,
            UncertAbs_S, criteria_UncertAbs_S_avg, criteria_UncertAbs_S_std,
            UncertRel_S, criteria_UncertRel_S_avg, criteria_UncertRel_S_std
        )

        if inputs.rank == 0:
            MD_step_index += 1
            # Acceptance check with criteria
            ##!! Epot_step should be rechecked.
            if random.random() < criteria and Epot_step > 0.1:
                accept = 'Accepted'
                MD_index += 1
                write_traj.write(atoms=struc_step)
            else:
                accept = 'Vetoed'

            # Record the MD results at the current step
            trajfile = open(f'UNCERT/uncertainty-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}.txt', 'a')
            trajfile.write(
                '{:.5e}'.format(Decimal(str(struc_step.get_temperature()))) + '\t' +
                uncert_strconvter(UncertAbs_E) + '\t' +
                uncert_strconvter(UncertRel_E) + '\t' +
                uncert_strconvter(UncertAbs_F) + '\t' +
                uncert_strconvter(UncertRel_F) + '\t' +
                uncert_strconvter(UncertAbs_S) + '\t' +
                uncert_strconvter(UncertRel_S) + '\t' +
                uncert_strconvter(Epot_step) + '\t' +
                uncert_strconvter(S_step) + '\t' +
                str(MD_index) + '          \t' +
                '{:.5e}'.format(Decimal(str(criteria))) + '\t' +
                str(accept) + '   \n'
            )
            trajfile.close()
        MD_index = inputs.comm.bcast(MD_index, root=0)

    mpi_print(f'[MLMD_main]\tThe main MLMD of the iteration {inputs.index} at {inputs.temperature}K and {inputs.pressure}bar is done', inputs.rank)


def traj_fromRealE(temperature, pressure, E_ref, E_gs, uncert_type, al_type, ntotal, index):
    """Function [traj_fromRealE]
    Get a configuratio nwith the largest real error
    to accelerate the investigation on unknown configurational space

    Parameters:

    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    E_ref: flaot
        The energy of reference state (Here, ground state)
    index: int
        The index of AL interactive step

    Returns:

    struc: ASE atoms
        A configuration with the largest real error
    """

    ##!! We need to develop the method to check 
    ##!! real error of not only energy but also force or force_max
    ##!! But, it is pratically not easy to do that
    ##!! since we need to store all force information.
    # Read the uncertainty file
    uncertainty_data = pd.read_csv(
        f'UNCERT/uncertainty-{temperature}K-{pressure}bar_{index-1}.txt',
        index_col=False, delimiter='\t'
        )

    if uncert_type == 'absolute':
        uncert_piece = 'Abs'
    elif uncert_type == 'relative':
        uncert_piece = 'Rel'

    if al_type == 'energy':
        al_piece = 'E'
    elif al_type == 'force' or 'force_max':
        al_piece = 'F'
    elif al_type == 'sigma' or 'sigma_max':
        al_piece = 'S'

    # Extract the accepted samples
    RealError_data = uncertainty_data[
        uncertainty_data['Acceptance'] == 'Accepted   '
    ]

    uncert_result = np.array(RealError_data['Uncert'+uncert_piece+'_'+al_piece])
    sorted_indices = np.argsort(uncert_result)
    first_ten_indices = sorted_indices[ntotal*(-1):][::-1]

    MLIP_energy = np.array(RealError_data['Epot_average'])[first_ten_indices]

    # Find the configuration with the largest real error
    max_RealError = 0
    max_index = 0
    for jndex in range(ntotal):
        atoms, atoms_potE, atoms_forces = read_aims(
            f'CALC/{temperature}K-{pressure}bar_1/{jndex}/aims/calculations/aims.out'
            )
        RealError = np.absolute(
            MLIP_energy[jndex] + E_ref - atoms_potE
            )
        if RealError > max_RealError:
            max_RealError = RealError
            max_index = jndex
    
    # Open the sampled trajectory file
    traj = Trajectory(
        f'TRAJ/traj-{temperature}K-{pressure}bar_{index}.traj',
        properties=['forces', 'velocities', 'temperature']
        )
    # Load the corresponding configuration
    struc = traj[max_index]
    
    return struc
    

    
def MLMD_random(
    inputs, kndex, steps_random, calculator, E_ref
):
    """Function [MLMD_random]
    Initiate the Molecular Dynamics with trained model
    to randomly sample the configuration

    Parameters:

    kndex: int
        The index for MLMD_initial steps
        It is used to resume a terminated calculation
    index: int
        The index of AL interactive step
    ensemble: str
        Type of MD ensembles; 'NVTLangevin'
    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    timestep: float
        The step interval for printing MD steps

    friction: float
        Strength of the friction parameter in NVTLangevin ensemble
    compressibility: float
        compressibility in units of eV/Angstrom**3 in NPTBerendsen
    taut: float
        Time constant for Berendsen temperature coupling
        in NVTBerendsen and NPT Berendsen
    taup: float
        Time constant for Berendsen pressure coupling in NPTBerendsen
    mask: Three-element tuple
        Dynamic elements of the computational box (x,y,z);
        0 is false, 1 is true

    loginterval: int
        The step interval for printing MD steps
    steps_random: int
        the length of MD run for random sampling

    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization
    calculator: ASE calculator
        Calculators from trained models
    E_ref: flaot
        The energy of reference state (Here, ground state)

    Returns:

    struc: ASE atoms
        A last configuration in the trajectory file
    """

    # When this initialization starts from scratch,
    if kndex == 0:
        # Even when it is very first iterative step,
        if inputs.index == 0:
            # Read the trajectory file from 'trajectory_train.son'
            metadata, traj = son.load('trajectory_train.son')
            traj_ther = traj[-1]
            # Convert 'trajectory.son' format to ASE atoms
            struc_step = Atoms(
                [atomic_numbers[item[1]] for item in traj[-1]['atoms']['symbols'] for inputs.index in range(item[0])],
                positions = traj[-1]['atoms']['positions'],
                cell = traj[-1]['atoms']['cell'],
                pbc = traj[-1]['atoms']['pbc'],
                velocities = traj[-1]['atoms']['velocities']
            )
        else: # When it has the pervious step,
            # Read the trajectory from previous trajectory file
            traj_init     = f'TRAJ/traj-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}.traj'
            traj_previous = Trajectory(traj_init, properties=\
                                       ['forces', 'velocities', 'temperature'])
            # Resume the MD calculation from last configuration in the trajectory file
            struc_step    = traj_previous[-1]; del traj_previous;
    else: # If it starts from terminated point,
        struc_step = []
        if inputs.rank == 0:
            # Read the trajectory from previous file
            traj_previous = Trajectory(
                f'TEMPORARY/temp-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}.traj',
                properties=['forces', 'velocities', 'temperature']
                )
            struc_step    = traj_previous[-1]; del traj_previous;
        # Resume the MD calculatio nfrom last configuration
        struc_step = inputs.comm.bcast(struc_step, root=0)
    
    # Implement MD calculation as long as steps_random
    if inputs.rank == 0:
        check_mkdir('TRAJ')
    logfile      = f'TRAJ/traj-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index+1}.log'
    trajectory   = f'TRAJ/traj-{inputs.temperature}K-{inputs.pressure}bar_{inputs.index+1}.traj'
    runMD(
        inputs, struc_step, inputs.steps_random,
        logfile, trajectory, calculator,
        signal_uncert=False, signal_append=False
    )
    
    mpi_print(
        f'[MLMD_rand] The MLMD with the random sampling of the iteration {inputs.index}'
        + f'at {inputs.temperature}K and {inputs.pressure}bar is done', inputs.rank
    )
