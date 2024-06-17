from ase.io.trajectory import Trajectory
from ase.io.trajectory import TrajectoryWriter
from ase.io.bundletrajectory import BundleTrajectory

import os
from vibes import son
import copy
import random
import numpy as np
import pandas as pd
from ase import Atoms
from decimal import Decimal
from ase.build import make_supercell
from ase.io import read as atoms_read
from ase.data   import atomic_numbers
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from libs.lib_util    import check_mkdir, single_print, read_aims
from libs.lib_md       import cont_runMD, runMD
from libs.lib_criteria import eval_uncert, uncert_strconvter, get_criteria, get_criteria_prob


def MLMD_main(
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

    condition = f'{inputs.temperature}K-{inputs.pressure}bar'

    # Extract the criteria information from the initialization step
    criteria_collected = get_criteria(inputs.temperature, inputs.pressure, inputs.index, inputs.steps_init, inputs.al_type)

    # Open a trajectory file to store the sampled configurations
    check_mkdir('TEMPORARY')
    check_mkdir('TRAJ')
    write_traj = TrajectoryWriter(
        filename=f'TRAJ/traj-{condition}_{inputs.index+1}.traj',
        mode='a'
        )

    # summary_msg = f'[MLMD] Calculate it from index {inputs.index}, MD_index {MD_index}'
    # summary_msg += f', MD_step_index {MD_step_index}' if inputs.calc_type == 'period' else ''
    single_print(f'[MLMD] Calculate it from index {inputs.index}, MD_index {MD_index}, MD_step_index {MD_step_index}')
    # When this initialization starts from scratch,
    if MD_index == 0:
        # Even when it is very first iterative step,
        if inputs.index != 0:
            # Name of the pervious uncertainty file
            uncert_file = f'UNCERT/uncertainty-{condition}_{inputs.index-1}.txt'

            # Check the existence of the file
            if os.path.exists(uncert_file):
                # Start from the configuration with the largest real error
                if inputs.output_format == 'nequip':
                    traj_temp     = f'TRAJ/traj-{condition}_{inputs.index}.traj'
                    single_print(f'[MLMD] Read a configuration from traj file')
                    # Read the trajectory from previous trajectory file
                    traj_previous = Trajectory(traj_temp, properties=\
                                               ['forces', 'velocities', 'temperature'])
                    # Resume the MD calculation from last configuration in the trajectory file
                    struc_step    = traj_previous[-1]; del traj_previous;
                else:
                    struc_step = traj_fromRealE(inputs.temperature, inputs.pressure, inputs.E_gs, inputs.uncert_type, inputs.al_type, inputs.ntotal, inputs.index)
                
                # Open the uncertainty file for current step
                check_mkdir('UNCERT')
                uncert_file_next = f'UNCERT/uncertainty-{condition}_{inputs.index}.txt'
                trajfile = open(uncert_file_next, 'w')
                title = 'Temperature[K]\t'
                if inputs.ensemble[:3] == 'NPT':
                    title += 'Pressure[GPa]\t'
                title += 'UncertAbs_E\tUncertRel_E\tUncertAbs_F\tUncertRel_F'\
                        +'\tUncertAbs_S\tUncertRel_S\tEpot_average\tS_average'\
                        +'\tCounting\tProbability\tAcceptance\n'
                trajfile.write(title)
                trajfile.close()

            else: # If there is no privous uncertainty file
                # Read the trajectory from previous trajectory file
                single_print(f'[MLMD] Read a configuration from traj file')
                if inputs.ensemble[:3] == 'NPT':
                    traj_temp     = f'TEMPORARY/temp-{condition}_{inputs.index}.bundle'
                    traj_previous = BundleTrajectory(traj_temp)
                else:
                    traj_temp     = f'TEMPORARY/temp-{condition}_{inputs.index}.traj'
                    traj_previous = Trajectory(traj_temp)
                # Resume the MD calculation from last configuration in the trajectory file
                struc_step    = traj_previous[-1]; del traj_previous;

        elif os.path.exists('start.in'):
            single_print(f'[MLMD] Read a configuration from start.in')
            # Read the ground state structure with the primitive cell
            struc_init = atoms_read('start.in', format='aims')
            # Make it supercell
            struc_step = make_supercell(struc_init, inputs.supercell_init)
            MaxwellBoltzmannDistribution(struc_step, temperature_K=inputs.temperature*1.5, force_temp=True)

        elif os.path.exists('start.traj'):
            single_print(f'[runMD]\tFound the start.traj file. MD starts from this.')
            # Read the ground state structure with the primitive cell
            struc_init = Trajectory('start.traj')[-1]
            struc_step = make_supercell(struc_init, inputs.supercell_init)
            del struc_init
            try:
                struc_step.get_velocities()
            except AttributeError:
                MaxwellBoltzmannDistribution(struc_step, temperature_K=inputs.temperature*1.5, force_temp=True)

        elif os.path.exists('start.bundle'):
            from ase.io.bundletrajectory import BundleTrajectory
            single_print(f'[runMD]\tFound the start.bundle file. MD starts from this.')
            file_traj_read = BundleTrajectory(filename='start.bundle', mode='r')
            file_traj_read[0]; #ASE bug
            struc_init = file_traj_read[-1]
            struc_step = make_supercell(struc_init, inputs.supercell_init)
            del struc_init
            try:
                struc_step.get_velocities()
            except AttributeError:
                MaxwellBoltzmannDistribution(struc_step, temperature_K=inputs.temperature*1.5, force_temp=True)

        else:
            single_print(f'[MLMD] Read a configuration from trajectory_train.son')
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
        # Read the trajectory from previous trajectory file
        single_print(f'[MLMD] Read a configuration from temp file')
        if inputs.ensemble[:3] == 'NPT':
            traj_temp     = f'TEMPORARY/temp-{condition}_{inputs.index}.bundle'
            traj_previous = BundleTrajectory(traj_temp)
        else:
            traj_temp     = f'TEMPORARY/temp-{condition}_{inputs.index}.traj'
            traj_previous = Trajectory(traj_temp)
        struc_step = traj_previous[-1]; del traj_previous;

    if os.path.exists('start.in') and (inputs.ensemble == 'NVTLangevin_meta' or inputs.ensemble == 'NVTLangevin_temp' or inputs.ensemble == 'NVTLangevin_bias' or inputs.ensemble == 'NVTLangevin_bias_temp') and MD_index == 0:
        single_print(f'[MLMD] Read a configuration from start.in')
        # Read the ground state structure with the primitive cell
        struc_init = atoms_read('start.in', format='aims')
        # Make it supercell
        struc_step = make_supercell(struc_init, inputs.supercell_init)
        MaxwellBoltzmannDistribution(struc_step, temperature_K=inputs.temperature*1.5, force_temp=True)

    cont_runMD(inputs, struc_step, MD_index, MD_step_index, calc_MLIP, E_ref, signal_uncert=False, signal_append=False)

    single_print(f'[MLMD_main]\tThe main MLMD of the iteration {inputs.index} at {inputs.temperature}K and {inputs.pressure}bar is done')


def traj_fromRealE(temperature, pressure, E_gs, uncert_type, al_type, ntotal, index):
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

    if al_type == 'energy' or 'energy_max':
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
            MLIP_energy[jndex] - atoms_potE
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

    condition = f'{inputs.temperature}K-{inputs.pressure}bar'

    # When this initialization starts from scratch,
    if kndex == 0:
        # Even when it is very first iterative step,
        if inputs.index == 0:
            # Read the trajectory file from 'trajectory_train.son'
            metadata, traj = son.load('trajectory_train.son')
            traj_ther = traj[-1]
            # Convert 'trajectory.son' format to ASE atoms
            struc_step = Atoms(
                [atomic_numbers[item[1]] for item in traj[-1]['atoms']['symbols'] for jdx in range(item[0])],
                positions = traj[-1]['atoms']['positions'],
                cell = traj[-1]['atoms']['cell'],
                pbc = traj[-1]['atoms']['pbc'],
                velocities = traj[-1]['atoms']['velocities']
            )
        else: # When it has the pervious step,
            # Read the trajectory from previous trajectory file
            traj_init     = f'TRAJ/traj-{condition}_{inputs.index}.traj'
            traj_previous = Trajectory(traj_init, properties=\
                                       ['forces', 'velocities', 'temperature'])
            # Resume the MD calculation from last configuration in the trajectory file
            struc_step    = traj_previous[-1]; del traj_previous;
    else: # If it starts from terminated point,
        struc_step = []
        # Read the trajectory from previous file
        traj_previous = Trajectory(
            f'TEMPORARY/temp-{condition}_{inputs.index}.traj',
            properties=['forces', 'velocities', 'temperature']
            )
        struc_step    = traj_previous[-1]; del traj_previous;
        # Resume the MD calculatio nfrom last configuration
    
    # Implement MD calculation as long as steps_random
    check_mkdir('TRAJ')
    logfile      = f'TRAJ/traj-{condition}_{inputs.index+1}.log'
    trajectory   = f'TRAJ/traj-{condition}_{inputs.index+1}.traj'
    runMD(
        inputs, struc_step, inputs.steps_random,
        logfile, trajectory, calculator,
        signal_uncert=False, signal_append=False
    )

    single_print(
        f'[MLMD_rand] The MLMD with the random sampling of the iteration {inputs.index}'
        + f'at {inputs.temperature}K and {inputs.pressure}bar is done'
    )





# def MLMD_initial(
#     kndex, index, ensemble, temperature, pressure, timestep, friction,
#     compressibility, taut, taup, mask, loginterval, steps_init, nstep,
#     nmodel, calculator, E_ref, al_type, harmonic_F, anharmonic_F
# ):
#     """Function [MLMD_initial]
#     Initiate the Molecular Dynamics with trained model
#     to get the average and standard deviation
#     of uncertainty and total energies

#     Parameters:

#     kndex: int
#         The index for MLMD_initial steps
#         It is used to resume a terminated calculation
#     index: int
#         The index of AL interactive step
#     ensemble: str
#         Type of MD ensembles; 'NVTLangevin'
#     temperature: float
#         The desired temperature in units of Kelvin (K)
#     pressure: float
#         The desired pressure in units of eV/Angstrom**3
#     timestep: float
#         The step interval for printing MD steps

#     friction: float
#         Strength of the friction parameter in NVTLangevin ensemble
#     compressibility: float
#         compressibility in units of eV/Angstrom**3 in NPTBerendsen
#     taut: float
#         Time constant for Berendsen temperature coupling
#         in NVTBerendsen and NPT Berendsen
#     taup: float
#         Time constant for Berendsen pressure coupling in NPTBerendsen
#     mask: Three-element tuple
#         Dynamic elements of the computational box (x,y,z);
#         0 is false, 1 is true

#     loginterval: int
#         The step interval for printing MD steps
#     steps_init: int
#         The number of initialization MD steps
#         to get averaged uncertainties and energies

#     nstep: int
#         The number of subsampling sets
#     nmodel: int
#         The number of ensemble model sets with different initialization
#     calculator: ASE calculator
#         Calculators from trained models
#     E_ref: flaot
#         The energy of reference state (Here, ground state)
#     al_type: str
#         Type of active learning: 'energy', 'force', 'force_max'

#     Returns:

#     struc_step: ASE atoms
#         Last entry of the configuration from initial trajectory
#     """

#     # Extract MPI infos
#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()

#     # When this initialization starts from scratch,
#     if kndex == 0:
#         # Even when it is very first iterative step,
#         if index == 0:
#             if os.path.exists('start.in'):
#                 # Read the ground state structure with the primitive cell
#                 struc_init = atoms_read(self.MD_input, format='aims')
#                 # Make it supercell
#                 struc_step = make_supercell(struc_init, self.supercell_init)
#                 MaxwellBoltzmannDistribution(struc, temperature_K=self.temperature, force_temp=True)
#             else:
#                 # Read the trajectory file from 'trajectory_train.son'
#                 metadata, traj = son.load('trajectory_train.son')
#                 # Take the last configuration from 'trajectory_train.son'
#                 traj_ther = traj[-1]
#                 # Convert 'trajectory.son' format to ASE atoms
#                 struc_step = Atoms(
#                     [atomic_numbers[item[1]] for item in traj[-1]['atoms']['symbols'] for index in range(item[0])],
#                     positions = traj[-1]['atoms']['positions'],
#                     cell = traj[-1]['atoms']['cell'],
#                     pbc = traj[-1]['atoms']['pbc'],
#                     velocities = traj[-1]['atoms']['velocities']
#                 )
#         else: # When it has the pervious step,
#             # # Name of the pervious uncertainty file
#             # uncert_file = f'UNCERT/uncertainty-{temperature}K-{pressure}bar_{index-1}.txt'

#             # # Check the existence of the file
#             # if os.path.exists(uncert_file):
#             #     # Start from the configuration with the largest real error
#             #     struc_step = traj_fromRealE(temperature, pressure, E_ref, index)
                
#             # else: # If there is no privous uncertainty file
#             # Read the trajectory from previous trajectory file
#             traj_init     = f'TRAJ/traj-{temperature}K-{pressure}bar_{index}.traj'
#             traj_previous = Trajectory(traj_init, properties=\
#                                        ['forces', 'velocities', 'temperature'])
#             # Resume the MD calculation from last configuration in the trajectory file
#             struc_step    = traj_previous[-1]; del traj_previous;

#             # Open the uncertainty file for current step
#             if rank == 0:
#                 check_mkdir('UNCERT')
#                 uncert_file_next = f'UNCERT/uncertainty-{temperature}K-{pressure}bar_{index}.txt'
#                 trajfile = open(uncert_file_next, 'w')
#                 trajfile.write(
#                     'Temperature[K]\tUncertAbs_E\tUncertRel_E\tUncertAbs_F\tUncertRel_F'
#                     +'\tUncertAbs_S\tUncertRel_S\tEpot_average\tS_average'
#                     +'\tCounting\tProbability\tAcceptance\n'
#                 )
#                 trajfile.close()

#     else: # If it starts from terminated point,
#         struc_step = []
#         if rank == 0:
#             # Read the trajectory from previous file
#             traj_previous = Trajectory(
#                 f'TEMPORARY/temp-{temperature}K-{pressure}bar_{index}.traj',
#                 properties=['forces', 'velocities', 'temperature']
#                 )
#             struc_step = traj_previous[-1]; del traj_previous;
#         # Resume the MD calculatio nfrom last configuration
#         struc_step = comm.bcast(struc_step, root=0)
    
#     # Initiate the MD run starting from kndex until reaching steps_init
#     for jndex in range(kndex, steps_init):
#         # MD information for temporary steps
#         check_mkdir('TEMPORARY')
#         trajectory   = f'TEMPORARY/temp-{temperature}K-{pressure}bar_{index}.traj'

#         # Implement MD calculation for only one loginterval step
#         runMD(
#             struc=struc_step, ensemble=ensemble, temperature=temperature,
#             pressure=pressure, timestep=timestep, friction=friction,
#             compressibility=compressibility, taut=taut, taup=taup,
#             mask=mask, loginterval=loginterval, steps=loginterval,
#             nstep=nstep, nmodel=nmodel, E_ref=E_ref, al_type=al_type,
#             logfile=None, trajectory=trajectory, calculator=calculator,
#             harmonic_F=harmonic_F, anharmonic_F=anharmonic_F, signal_uncert=False, signal_append=False
#         )

#         # Get new configuration and velocities for next step
#         traj_current = Trajectory(
#             trajectory, properties=['forces', 'velocities', 'temperature']
#             )
#         struc_step   = traj_current[-1]
#         del traj_current # Remove it to reduce the memory usage

#         # Get absolute and relative uncertainties of energy and force
#         # and also total energy
#         UncertAbs_E, UncertRel_E, UncertAbs_F, UncertRel_F, Epot_step =\
#         eval_uncert(struc_step, nstep, nmodel, E_ref, calculator, al_type, harmonic_F)

#         # Record the all uncertainty and total energy information at the current step
#         if rank == 0:
#             trajfile = open(f'UNCERT/uncertainty-{temperature}K-{pressure}bar_{index}.txt', 'a')
#             trajfile.write(
#                 '{:.5e}'.format(Decimal(str(struc_step.get_temperature()))) + '\t' +
#                 uncert_strconvter(UncertAbs_E) + '\t' +
#                 uncert_strconvter(UncertRel_E) + '\t' +
#                 uncert_strconvter(UncertAbs_F) + '\t' +
#                 uncert_strconvter(UncertRel_F) + '\t' +
#                 '{:.5e}'.format(Decimal(str(Epot_step))) + '\t' +
#                 f'initial_{jndex+1}' +
#                 '\t--         \t--         \t\n'
#             )
#             trajfile.close()
    
#     mpi_print(
#         f'[MLMD_init]\tThe initial MLMD of the iteration {index}'
#         + f'at {temperature}K and {pressure}bar is done', rank
#     )
    
#     return struc_step
