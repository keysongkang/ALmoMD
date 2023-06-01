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
from ase.data   import atomic_numbers

from libs.lib_util    import mpi_print, single_print, read_aims
from libs.lib_md       import runMD
from libs.lib_criteria import eval_uncert, uncert_strconvter, get_criteria, get_criteria_prob



def MLMD_initial(
    kndex, index, ensemble, temperature, pressure, timestep, friction,
    compressibility, taut, taup, mask, loginterval, steps_init, nstep,
    nmodel, calculator, E_ref, al_type
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
            # Name of the pervious uncertainty file
            uncert_file = f'uncertainty-{temperature}K-{pressure}bar_{index-1}.txt'

            # Check the existence of the file
            if os.path.exists(uncert_file):
                # Start from the configuration with the largest real error
                struc_step = traj_fromRealE(temperature, pressure, E_ref, index)
                
                # Open the uncertainty file for current step
                if rank == 0:
                    uncert_file_next = f'uncertainty-{temperature}K-{pressure}bar_{index}.txt'
                    trajfile = open(uncert_file_next, 'w')
                    trajfile.write(
                        'Temp.[K]\tUncertRel_E\tUncertAbs_E\tUncertRel_F\tUncertAbs_F\t'
                        + 'Epot_average\tCounting\tProbability\tAcceptance\n'
                    )
                    trajfile.close()
            else: # If there is no privous uncertainty file
                # Read the trajectory from previous trajectory file
                traj_init     = f'traj-{temperature}K-{pressure}bar_{index}.traj'
                traj_previous = Trajectory(traj_init, properties=\
                                           ['forces', 'velocities', 'temperature'])
                # Resume the MD calculation from last configuration in the trajectory file
                struc_step    = traj_previous[-1]; del traj_previous;
    else: # If it starts from terminated point,
        struc_step = []
        if rank == 0:
            # Read the trajectory from previous file
            traj_previous = Trajectory(
                f'temp-{temperature}K-{pressure}bar_{index}.traj',
                properties=['forces', 'velocities', 'temperature']
                )
            struc_step = traj_previous[-1]; del traj_previous;
        # Resume the MD calculatio nfrom last configuration
        struc_step = comm.bcast(struc_step, root=0)
    
    # Initiate the MD run starting from kndex until reaching steps_init
    for jndex in range(kndex, steps_init):
        # MD information for temporary steps
        trajectory   = f'temp-{temperature}K-{pressure}bar_{index}.traj'

        # Implement MD calculation for only one loginterval step
        runMD(
            struc=struc_step, ensemble=ensemble, temperature=temperature,
            pressure=pressure, timestep=timestep, friction=friction,
            compressibility=compressibility, taut=taut, taup=taup,
            mask=mask, loginterval=loginterval, steps=loginterval,
            nstep=nstep, nmodel=nmodel, E_ref=E_ref, al_type=al_type,
            logfile=None, trajectory=trajectory, calculator=calculator,
            signal_uncert=False, signal_append=False
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
        eval_uncert(struc_step, nstep, nmodel, E_ref, calculator, al_type)

        # Record the all uncertainty and total energy information at the current step
        if rank == 0:
            trajfile = open(f'uncertainty-{temperature}K-{pressure}bar_{index}.txt', 'a')
            trajfile.write(
                '{:.5e}'.format(Decimal(str(struc_step.get_temperature()))) + '\t' +
                uncert_strconvter(UncertRel_E) + '\t' +
                uncert_strconvter(UncertAbs_E) + '\t' +
                uncert_strconvter(UncertRel_F) + '\t' +
                uncert_strconvter(UncertAbs_F) + '\t' +
                '{:.5e}'.format(Decimal(str(Epot_step))) + '\t' +
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
    struc_step, al_type, uncert_type
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
    Uncert_type: str
        Type of uncertainty; 'absolute', 'relative'
    """

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Extract the criteria information from the initialization step
    criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std,\
    criteria_UncertRel_E_avg, criteria_UncertRel_E_std,\
    criteria_UncertAbs_F_avg, criteria_UncertAbs_F_std,\
    criteria_UncertRel_F_avg, criteria_UncertRel_F_std,\
    criteria_Epot_step_avg, criteria_Epot_step_std\
    = get_criteria(temperature, pressure, index, steps_init)
    
    # Open a trajectory file to store the sampled configurations
    write_traj = TrajectoryWriter(
        filename=f'traj-{temperature}K-{pressure}bar_{index+1}.traj',
        mode='a'
        )

    # When it resumes the calculation,
    if MD_index != 0:
        if rank == 0:
            # Read the trajectory from previous file
            traj_temp = Trajectory(
                f'temp-{temperature}K-{pressure}bar_{index}.traj',
                properties='energy, forces'
                )
            struc_step = traj_temp[-1]; del traj_temp;
        # Resume the MD calculatio nfrom last configuration
        struc_step = comm.bcast(struc_step, root=0)
    
    # Initiate the MD run starting from MD_index until reaching ntotal
    # MD_index also indicates the number of sampled configurations
    while MD_index < ntotal:
        accept = '--         '

        # MD information for temporary steps
        trajectory   = f'temp-{temperature}K-{pressure}bar_{index}.traj'
        
        # Implement the MD calculation for only one loginterval step
        runMD(
            struc_step, ensemble=ensemble, temperature=temperature,
            pressure=pressure, timestep=timestep, friction=friction,
            compressibility=compressibility, taut=taut, taup=taup,
            mask=mask, loginterval=loginterval, steps=loginterval,
            nstep=nstep, nmodel=nmodel, E_ref=E_ref, al_type=al_type,
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
        UncertAbs_E, UncertRel_E, UncertAbs_F, UncertRel_F, Epot_step =\
        eval_uncert(struc_step, nstep, nmodel, E_ref, calc_MLIP, al_type)
        
        # Get a criteria probability from uncertainty and energy informations
        criteria = get_criteria_prob(
            al_type, uncert_type, kB, NumAtoms, temperature,
            Epot_step, criteria_Epot_step_avg, criteria_Epot_step_std,
            UncertAbs_E, criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std,
            UncertRel_E, criteria_UncertRel_E_avg, criteria_UncertRel_E_std,
            UncertAbs_F, criteria_UncertAbs_F_avg, criteria_UncertAbs_F_std,
            UncertRel_F, criteria_UncertRel_F_avg, criteria_UncertRel_F_std
        )
        
        if rank == 0:
            # Acceptance check with criteria
            ##!! Epot_step should be rechecked.
            if random.random() < criteria and Epot_step > 0.1:
                accept = 'Accepted'
                MD_index += 1
                write_traj.write(atoms=struc_step)
            else:
                accept = 'Vetoed'

            # Record the MD results at the current step
            trajfile = open(f'uncertainty-{temperature}K-{pressure}bar_{index}.txt', 'a')
            trajfile.write(
                '{:.5e}'.format(Decimal(str(struc_step.get_temperature()))) + '\t' +
                uncert_strconvter(UncertRel_E) + '\t' +
                uncert_strconvter(UncertAbs_E) + '\t' +
                uncert_strconvter(UncertRel_F) + '\t' +
                uncert_strconvter(UncertAbs_F) + '\t' +
                uncert_strconvter(Epot_step) + '\t' +
                str(MD_index) + '          \t' +
                '{:.5e}'.format(Decimal(str(criteria))) + '\t' +
                str(accept) + '   \n'
            )
            trajfile.close()
        MD_index = comm.bcast(MD_index, root=0)

    mpi_print(f'The main MLMD of the iteration {index} at {temperature}K and {pressure}bar is done', rank)



def traj_fromRealE(temperature, pressure, E_ref, index):
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
        f'uncertainty-{temperature}K-{pressure}bar_{index-1}.txt',
        index_col=False, delimiter='\t'
        )

    # Extract the accepted samples
    RealError_data = uncertainty_data.loc[
        uncertainty_data['Acceptance'] == 'Accepted   '
    ]

    # Find the configuration with the largest real error
    max_RealError = 0
    max_index = 0
    for jndex in range(len(RealError_data)):
        atoms, atoms_potE, atoms_forces = read_aims(
            f'calc/{temperature}K-{pressure}bar_1/{jndex}/aims/calculations/aims.out'
            )
        RealError = np.absolute(
            np.array(RealError_data['Epot_average'])[jndex] + E_ref - atoms_potE
            )
        if RealError > max_RealError:
            max_RealError = RealError
            max_index = jndex
    
    # Open the sampled trajectory file
    traj = Trajectory(
        f'traj-{temperature}K-{pressure}bar_{index}.traj',
        properties=['forces', 'velocities', 'temperature']
        )
    # Load the corresponding configuration
    struc = traj[max_index]
    
    return struc
    

    
def MLMD_random(
    kndex, index, ensemble, temperature, pressure, timestep, friction,
    compressibility, taut, taup, mask, loginterval, steps_random, nstep,
    nmodel, calculator, E_ref
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

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # When this initialization starts from scratch,
    if kndex == 0:
        # Even when it is very first iterative step,
        if index == 0:
            # Read the trajectory file from 'trajectory_train.son'
            metadata, traj = son.load('trajectory_train.son')
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
            # Read the trajectory from previous trajectory file
            traj_init     = f'traj-{temperature}K-{pressure}bar_{index}.traj'
            traj_previous = Trajectory(traj_init, properties=\
                                       ['forces', 'velocities', 'temperature'])
            # Resume the MD calculation from last configuration in the trajectory file
            struc_step    = traj_previous[-1]; del traj_previous;
    else: # If it starts from terminated point,
        struc_step = []
        if rank == 0:
            # Read the trajectory from previous file
            traj_previous = Trajectory(
                f'temp-{temperature}K-{pressure}bar_{index}.traj',
                properties=['forces', 'velocities', 'temperature']
                )
            struc_step    = traj_previous[-1]; del traj_previous;
        # Resume the MD calculatio nfrom last configuration
        struc_step = comm.bcast(struc_step, root=0)
    
    # Implement MD calculation as long as steps_random
    logfile      = f'traj-{temperature}K-{pressure}bar_{index+1}.log'
    trajectory   = f'traj-{temperature}K-{pressure}bar_{index+1}.traj'
    runMD(
        struc=struc_step, ensemble=ensemble, temperature=temperature,
        pressure=pressure, timestep=timestep, friction=friction,
        compressibility=compressibility, taut=taut, taup=taup,
        mask=mask, loginterval=loginterval, steps=steps_random,
        nstep=nstep, nmodel=nmodel, E_ref=E_ref, al_type=al_type,
        logfile=logfile, trajectory=trajectory, calculator=calculator,
        signal_uncert=False, signal_append=False
    )
    
    # Recond the index of the iterative steps
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