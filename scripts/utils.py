import os
import re
import sys
from vibes import son
import random
import argparse
import collections
import numpy as np
import ase.units as units
from ase import Atoms
from ase.data import atomic_numbers
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from libs.lib_util import check_mkdir, rm_file, single_print, output_init

version = '0.0.0'

def aims2son(temperature):
    """Function [aims2son]
    Convert aims.out to trajectory.son and assign the velocity based on the 
    Maxwell-Boltzmann distribution using the temperature.
    
    Currently it works with FHI-aims version 220506.

    Parameters:

    temperature: float
        temperature in units of Kelvin.
    """

    # Print the head
    output_init('aims2son', version, rank=0)
    single_print(f'[aims2son]\tConvert aims.out to trajectory.son')

    # Tracking the reading line for the specific contents
    index_struc = 0             # Read a structral information
    index_force = 0             # Read force components
    index_stress_whole = 0      # Read stress tensors for unit cell
    index_stress_individual = 0 # Read stress tensors for each atom
    signal = 0                  # Indicate each step

    # Properties to be read
    cell = []
    forces = []
    numbers = []
    numbers_symbol = []
    positions = []
    mass = []
    stress_whole = []
    stress_individual = []
    pbc = [True, True, True]    # Currently, always periodic 
    NumAtoms = 0

    single_print(f'[aims2son]\tRead aims.out file ...')
    with open('aims.out', "r") as file_one: # Open aims.out file
        for line in file_one: # Go through whole contents line by line

            # Assign NumAoms by searching "Number of atoms"
            if re.search('Number of atoms', line):
                NumAtoms = int(re.findall(r'\d+', line)[0])

            # Assign mass by searching "Found atomic mass"
            # It follows the sequence of basis info
            if re.search('Found atomic mass :', line):
                mass.append(float(float(re.findall(r'[+-]?\d+(?:\.\d+)?', line)[0])))

            # Assign total_E by searching "Total energy corrected"
            if re.search('Total energy corrected', line):
                total_E = float(re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', line)[0])

            # Assign structural information by searching "Atomic structure ..."
            if re.search('Atomic structure that was used in the preceding time step of the wrapper', line):
                index_struc = 1  # indicates it found that line
            # When it found that,
            if index_struc > 0:
                # Assign cell (lattice parameters) after 3 lines
                if index_struc > 2 and index_struc < 6:
                    cell.append([float(i) for i in re.findall(r'[+-]?\d+(?:\.\d+)?', line)])
                    index_struc += 1
                # Assign (atomic) positions after 7 lines until 7+NumAtoms lines
                elif index_struc > 6 and index_struc < (7+NumAtoms):
                    positions.append([float(i) for i in re.findall(r'[+-]?\d+(?:\.\d+)?', line)])
                    numbers.append(atomic_numbers[(line[-3:].replace(' ', '')).replace('\n','')])
                    numbers_symbol.append((line[-3:].replace(' ', '')).replace('\n',''))
                    index_struc += 1
                # When it reaches 7+NumAtoms line, 
                elif index_struc == (7+NumAtoms):
                    index_struc = 0  # initilize index_struc
                    signal = 1       # indicates that current MD step is done
                else:
                    index_struc += 1 # Otherwise, skip a line

            # Assign forces (on atoms) by searching "Total atomic forces"
            if re.search('Total atomic forces', line):
                index_force = 1  # indicates it found that line
            # When it found that,
            if index_force > 0:
                # Assign forces on atoms after 2 lines until 2+NumAtoms lines
                if index_force > 1 and index_force < (2+NumAtoms):
                    forces.append([float(i) for i in re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', line)])
                    index_force += 1
                # When it reaches 2+NumAtoms line,
                elif index_force == (2+NumAtoms):
                    index_force = 0  # initialize index_force
                else:
                    index_force += 1 # Otherwise, skip a line

            # Assign stress_whole (stess tensor on unitcell) by searching "Analytical stress tensor"
            if re.search('Analytical stress tensor - Symmetrized', line):
                index_stress_whole = 1 # indicates it found that line
            # When it found that,
            if index_stress_whole > 0:
                # Assign stress tensor after 6 lines
                if index_stress_whole > 5 and index_stress_whole < 9:
                    stress_whole.append([float(i) for i in re.findall(r'[+-]?\d+(?:\.\d+)?', line)])
                    index_stress_whole += 1
                # When it reaches 6+3 line,
                elif index_stress_whole == 9:
                    index_stress_whole = 0  # initialize index_stress_whole
                else:
                    index_stress_whole += 1 # Otherwise, skip a line

            # Assign stress_individual (stress tensor on each atom) by searching "used for heat flux"
            if re.search('used for heat flux calculation', line):
                index_stress_individual = 1 # indicates it found that line
            # When it found that,
            if index_stress_individual > 0:
                # Assign stress tensor for each atom after 4 lines until 4+NumAtoms lines
                if index_stress_individual > 3 and index_stress_individual < (4+NumAtoms):
                    stress_temp = [float(i) for i in re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', line)]
                    stress_individual.append([[stress_temp[0],stress_temp[4],stress_temp[5]],[stress_temp[4],stress_temp[2],stress_temp[5]],[stress_temp[5],stress_temp[5],stress_temp[3]]])
                    index_stress_individual += 1
                # When it reaches 4+NumAtoms line,
                elif index_stress_individual == 4+NumAtoms:
                    index_stress_individual = 0  # initialize index_stres_individual
                else:
                    index_stress_individual += 1 # Otherwise, skip a line

            # End of each MD step
            if signal:
                # Create the ASE atoms
                atom = Atoms(
                    numbers,
                    positions=positions,
                    cell=cell,
                    pbc=pbc
                )
                # Assign the velocities at target temperature based on the Maxwell-Boltzmann distribution
                MaxwellBoltzmannDistribution(atom, temperature_K=temperature, force_temp=True)

                # Convert the format of symbols and masses for trajectory.son
                symbols = []
                masses = []
                idx = 0
                # key: atomic element, value: the total number of coressponding atoms
                for key, value in collections.Counter(numbers_symbol).items():
                    symbols.append([value, key])
                    masses.append([value, mass[idx]])
                    idx += 1

                # Prepare the dictionary to summarize the atoms_info
                atoms_info = {
                    "pbc": pbc,
                    "cell": cell,
                    "positions": positions,
                    "velocities": atom.get_velocities().tolist(),
                    "symbols": symbols,
                    "masses": masses
                }

                # Prepare the dictionary to summarize the calculator_info
                calculator_info = {
                    "energy": total_E,
                    "forces": forces,
                    "stress": stress_whole,
                    "stresses": stress_individual
                }

                # Merge the dictionaries together
                atom_dict = {
                    "atoms": atoms_info,
                    "calculator": calculator_info
                }

                # Initialize all indexes for next MD step
                index_struc = 0
                index_force = 0
                index_stress_whole = 0
                index_stress_individual = 0
                signal = 0

                # Initialize all properties for next MD step
                cell = []
                forces = []
                numbers = []
                numbers_symbol = []
                positions = []
                stress_whole = []
                stress_individual = []

                # Dump extracted atom info summary of current MD step into trajectory.son
                son.dump(atom_dict, 'trajectory.son', is_metadata=False)

    single_print(f'[aims2son]\t!! Finish converting aims.out to trajectory.son')


def split_son(num_split, E_gs, harmonic_F=False):
    """Function [split_son]
    Separate all trajectory file (trajectory.son)
    into testing data (trajectory_test.son, and data-test.npz)
    and training data (trajectory_train.son).

    Parameters:

    num_split: int
        The number of testing data.
    E_gs: float
        Reference total energy in units of eV/Unitcell
        to shift the total energies of the trajectory
        and avoid unusually high total energy values
        that may lead to unusual weightings with force values.
        Recommend to use the ground state total energy.
    """
    from libs.lib_util     import eval_sigma

    # Print the head
    output_init('split_son', version, rank=0)
    single_print(f'[split_son]\tInitiate splitting trajectory.son')

    if harmonic_F:
        single_print(f'[split_son]\tharmoic_F = True: Harmonic term will be excluded')

    single_print(f'[split_son]\tRead trajectory.son file')
    # Read trajectory.son file
    metadata, data = son.load('trajectory.son')

    # Randomly sample testing data with a total count of num_split.
    test_data = random.sample(data, num_split)
    # Extract the training data that is not included in the testing data
    train_data = [d for d in data if d not in test_data]
    
    # Check the existance of trajectory_test.son and trajectory_train.son files,
    # because it is annoying when we mixuse these files with different sampling
    if os.path.exists('trajectory_test.son') and os.path.exists('trajectory_train.son'):
        single_print('[split_son]\tThe files trajectory_test.son and trajectory_train.son already exist.')
        # May need to prepare for the conversion of SON file to NPZ file
        metadata, test_data = son.load('trajectory_test.son')
    else:
        single_print('[split_son]\tCollect samples for training and testing data.')
        rm_file('trajectory_test.son')
        rm_file('trajectory_train.son')
        rm_file('MODEL/data-test.npz')
        if metadata is not None:
            son.dump(metadata, 'trajectory_test.son', is_metadata=True)
            son.dump(metadata, 'trajectory_train.son', is_metadata=True)
        for test_item in test_data:
            son.dump(test_item, 'trajectory_test.son')   # Save testing data into trajectory_test.son file
        for train_item in train_data:
            son.dump(train_item, 'trajectory_train.son') # Save training data into trajectory_train.son file

    ### Save testing data in form of npz file for validation error checking.
    # Create a folder named data
    check_mkdir('MODEL')

    # Check if the data-test.npz file exists
    if os.path.exists('MODEL/data-test.npz'):
        single_print('[split_son]\tThe file data-test.npz already exists.')
    else:
        # Prepare the empty lists for properties
        E_test      = [] # Total energy
        F_test      = [] # Forces
        R_test      = [] # Atomic positions
        z_test      = [] # Chemical elements
        CELL_test   = [] # Lattice paratmers
        PBC_test    = [] # Periodicity
        sigma_test  = [] # Anharmonicity

        if harmonic_F:
            E_test_ori = []
            F_test_ori = []

        # Dump the informations
        for test_item in test_data:
            # E_test.append(test_item['calculator']['energy'] - E_gs); # Shift the total energy by the reference value.

            if harmonic_F:
                from libs.lib_util import get_displacements, get_fc_ha, get_E_ha
                displacements = get_displacements(test_item['atoms']['positions'], 'geometry.in.supercell')
                F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
                F_step = np.array(test_item['calculator']['forces']) - F_ha
                E_ha = get_E_ha(displacements, F_ha)
                E_step = np.array(test_item['calculator']['energy']) - E_gs - E_ha
                F_step_ori = np.array(test_item['calculator']['forces'])
                E_step_ori = np.array(test_item['calculator']['energy']) - E_gs
            else:
                F_step = np.array(test_item['calculator']['forces'])
                E_step = np.array(test_item['calculator']['energy']) - E_gs

            E_test.append(E_step);
            F_test.append(F_step);

            if harmonic_F:
                E_test_ori.append(E_step_ori)
                F_test_ori.append(F_step_ori)

            R_test.append(test_item['atoms']['positions']);
            z_test.append([atomic_numbers[item[1]] for item in test_item['atoms']['symbols'] for index in range(item[0])]); # Convert format
            CELL_test.append(test_item['atoms']['cell']);
            PBC_test.append(test_item['atoms']['pbc'])
            sigma_test.append(
                eval_sigma(
                    struc_step_forces = test_item['calculator']['forces'],
                    struc_step_positions = test_item['atoms']['positions'],
                    al_type = 'sigma'
                    )
                )
        
        # Save all information into data-test.npz
        npz_name = 'MODEL/data-test.npz'
        np.savez(
            npz_name[:-4],
            E=np.array(E_test),
            F=np.array(F_test),
            R=np.array(R_test),
            z=np.array(z_test),
            CELL=np.array(CELL_test),
            PBC=np.array(PBC_test),
            sigma=np.array(sigma_test)
        )
        
        if harmonic_F:
            # Save all information into data-test.npz
            npz_name = 'MODEL/data-test_ori.npz'
            np.savez(
                npz_name[:-4],
                E=np.array(E_test_ori),
                F=np.array(F_test_ori),
                R=np.array(R_test),
                z=np.array(z_test),
                CELL=np.array(CELL_test),
                PBC=np.array(PBC_test),
                sigma=np.array(sigma_test)
            )

        single_print('[split_son]\tFinish the sampling testing data: data-train.npz')

    single_print('[split_son]\t!! Finish the splitting process')


def harmonic_run(temperature, num_sample, DFT_calc, num_calc):
    """Frunction [harmonic_run]
    Initiate FHI-aims and FHI-vibes with structral configurations
    from a harmonic sampling

    Parameters:

    temperature: float
        Temperature (K)
    num_sample: int
        The number of harmonic samples
    DFT_calc: str
        The name of the DFT calculator
    num_calc: int
        The number of job scripts to be submitted
    """
    import subprocess
    from ase.io.aims import read_aims
    from ase.io.aims import write_aims

    # Print the head
    output_init('harmo_run', version, rank=0)
    single_print(f'[harmo_run]\tInitiate DFT calc. with harmonic samples')

    # Prepare the index inputs
    index_temp = '{:0>4.0f}'.format(temperature)
    index_calc_list = [f'{i:03d}' for i in range(num_sample)]

    # Create the calculation directory
    check_mkdir(f'raw')

    # Get the current path
    mainpath_cwd = os.getcwd()

    # Move to the calculation directory
    os.chdir(f'raw')

    # Get the full path to the calculation directotry
    calcpath_cwd = os.getcwd()

    single_print(f'[harmo_run]\tGet the template of the inputs')
    # Get the template of the job script
    with open('../DFT_INPUTS/job.slurm', 'r') as job_script_DFT_initial:
        job_script_DFT_default = job_script_DFT_initial.read()
    # Prepare the command line for FHI-aims for DeepH or FHI-vibes
    if DFT_calc == 'aims':
        DFT_command = 'srun /u/kkang/programs/FHIaims-DeepH/build2/aims.220609.scalapack.mpi.x > aims.out 2>&1'
    elif DFT_calc == 'vibes':
        DFT_command = 'vibes run singlepoint aims.in &> log.aims'

    # Prepare an empty list for the calculation paths
    execute_cwd = []

    single_print(f'[harmo_run]\tDeploy all DFT calculations')
    for idx in range(num_sample):
        # Create a folder for each structral configuration
        check_mkdir(f'{idx}')
        # Move to that folder
        os.chdir(f'{idx}')

        if DFT_calc == 'aims':
            if os.path.exists(f'aims.out'):
                # Check whether calculation is finished
                if 'Have a nice day.' in open('aims.out').read():
                    os.chdir(calcpath_cwd)
                else:
                    # Collect the current calculation path
                    execute_cwd.append(os.getcwd())
                    # Move back to 'calc' folder
                    os.chdir(calcpath_cwd)
            else:
                # Copy a configuration from the harmonic sampling
                harmonic_file = f'geometry.in.supercell.{index_temp}K.{index_calc_list[idx]}'
                # Need to convert harmonic_file in scaled positions for DeepH
                harmonic_atom = read_aims(f'./../../HARMONIC/{harmonic_file}')
                write_aims('geometry.in', harmonic_atom, scaled=True)
                # Get FHI-aims inputs from the template folder
                subprocess.run(['cp', './../../DFT_INPUTS/control.in', '.'])
                # Collect the current calculation path
                execute_cwd.append(os.getcwd())
                # Move back to 'calc' folder
                os.chdir(calcpath_cwd)
        elif DFT_calc == 'vibes':
            if os.path.exists(f'aims/calculations/aims.out'):
                # Check whether calculation is finished
                if 'Have a nice day.' in open('aims/calculations/aims.out').read():
                    os.chdir(calcpath_cwd)
                else:
                    # Collect the current calculation path
                    execute_cwd.append(os.getcwd())
                    # Move back to 'calc' folder
                    os.chdir(calcpath_cwd)
            else:
                # Copy a configuration from the harmonic sampling
                harmonic_file = f'geometry.in.supercell.{index_temp}K.{index_calc_list[idx]}'
                # Need to convert harmonic_file in scaled positions for DeepH
                harmonic_atom = read_aims(f'./../../HARMONIC/{harmonic_file}')
                write_aims('geometry.in', harmonic_atom, scaled=True)
                # subprocess.run(['cp', f'./../../HARMONIC/{harmonic_file}', 'geometry.in'])
                # Get FHI-aims inputs from the template folder
                subprocess.run(['cp', './../../DFT_INPUTS/aims.in', '.'])
                # Collect the current calculation path
                execute_cwd.append(os.getcwd())
                # Move back to 'calc' folder
                os.chdir(calcpath_cwd)

    single_print(f'[harmo_run]\tSubmit all job scripts for all DFT calculations')
    # Create job scripts and submit them
    for index_calc in range(num_calc):
        job_script = f'job_{index_calc}.slurm'
        with open(job_script, 'w') as writing_input:
            writing_input.write(job_script_DFT_default)
            for index_execute_cwd, value_execute_cwd in enumerate(execute_cwd):
                if index_execute_cwd % num_calc == index_calc:
                    writing_input.write('cd '+value_execute_cwd+'\n')
                    writing_input.write(DFT_command+'\n')
        # If the previous calculation is not finished, rerun it
        subprocess.run(['sbatch', job_script])

    # Move back to the original position
    os.chdir(mainpath_cwd)
    single_print(f'[harmo_run]\t!! Finish the DFT calculations with harmonic samples')


def harmonic2son(temperature, num_sample, output_format):
    """Frunction [harmonic_run]
    Collect all results of FHI-vibes calculations
    with harmonic samplings and convert them to SON file

    Parameters:

    temperature: float
        Temperature (K)
    num_sample: int
        The number of harmonic samples
    """

    # Print the head
    output_init('harmo2son', version, rank=0)
    single_print(f'[harmo2son]\tCollect all DFT results and convert them to SON file')

    # Prepare the index inputs
    index_temp = '{:0>4.0f}'.format(temperature)
    index_calc_list = [f'{i:03d}' for i in range(num_sample)]

    # Get the current path
    mainpath_cwd = os.getcwd()

    # Move to the calculation directory
    os.chdir(f'raw')

    # Get the full path to the calculation directotry
    calcpath_cwd = os.getcwd()

    single_print(f'[harmo2son]\tGo through all DFT results')
    for idx in range(num_sample):
        # Create a folder for each structral configuration
        print(f'Sample {idx}')
        check_mkdir(f'{idx}')
        # Move to that folder
        os.chdir(f'{idx}')

        if output_format == 'aims':
            if os.path.exists(f'aims/calculations/aims.out'):
                # Check whether calculation is finished
                if 'Have a nice day.' in open('aims/calculations/aims.out').read():
                    # Tracking the reading line for the specific contents
                    index_struc = 0             # Read a structral information
                    index_force = 0             # Read force components
                    index_stress_whole = 0      # Read stress tensors for unit cell
                    index_stress_individual = 0 # Read stress tensors for each atom
                    signal = 0                  # Indicate each step

                    # Properties to be read
                    cell = []
                    forces = []
                    numbers = []
                    numbers_symbol = []
                    positions = []
                    mass = []
                    stress_whole = []
                    stress_individual = []
                    pbc = [True, True, True]    # Currently, always periodic 
                    NumAtoms = 0

                    with open('aims/calculations/aims.out', "r") as file_one: # Open aims.out file
                        for line in file_one: # Go through whole contents line by line

                            # Assign NumAoms by searching "Number of atoms"
                            if re.search('Number of atoms', line):
                                NumAtoms = int(re.findall(r'\d+', line)[0])

                            # Assign mass by searching "Found atomic mass"
                            # It follows the sequence of basis info
                            if re.search('Found atomic mass :', line):
                                mass.append(float(float(re.findall(r'[+-]?\d+(?:\.\d+)?', line)[0])))

                            # Assign total_E by searching "Total energy corrected"
                            if re.search('Total energy corrected', line):
                                total_E = float(re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', line)[0])

                            # Assign structural information by searching "Atomic structure ..."
                            if re.search('Atomic structure that was used in the preceding time step of the wrapper', line):
                                index_struc = 1  # indicates it found that line
                            # When it found that,
                            if index_struc > 0:
                                # Assign cell (lattice parameters) after 3 lines
                                if index_struc > 2 and index_struc < 6:
                                    cell.append([float(i) for i in re.findall(r'[+-]?\d+(?:\.\d+)?', line)])
                                    index_struc += 1
                                # Assign (atomic) positions after 7 lines until 7+NumAtoms lines
                                elif index_struc > 6 and index_struc < (7+NumAtoms):
                                    positions.append([float(i) for i in re.findall(r'[+-]?\d+(?:\.\d+)?', line)])
                                    numbers.append(atomic_numbers[(line[-3:].replace(' ', '')).replace('\n','')])
                                    numbers_symbol.append((line[-3:].replace(' ', '')).replace('\n',''))
                                    index_struc += 1
                                # When it reaches 7+NumAtoms line, 
                                elif index_struc == (7+NumAtoms):
                                    index_struc = 0  # initilize index_struc
                                    signal = 1       # indicates that current MD step is done
                                else:
                                    index_struc += 1 # Otherwise, skip a line

                            # Assign forces (on atoms) by searching "Total atomic forces"
                            if re.search('Total atomic forces', line):
                                index_force = 1  # indicates it found that line
                            # When it found that,
                            if index_force > 0:
                                # Assign forces on atoms after 2 lines until 2+NumAtoms lines
                                if index_force > 1 and index_force < (2+NumAtoms):
                                    forces.append([float(i) for i in re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', line)])
                                    index_force += 1
                                # When it reaches 2+NumAtoms line,
                                elif index_force == (2+NumAtoms):
                                    index_force = 0  # initialize index_force
                                else:
                                    index_force += 1 # Otherwise, skip a line

                            # Assign stress_whole (stess tensor on unitcell) by searching "Analytical stress tensor"
                            if re.search('Analytical stress tensor - Symmetrized', line):
                                index_stress_whole = 1 # indicates it found that line
                            # When it found that,
                            if index_stress_whole > 0:
                                # Assign stress tensor after 6 lines
                                if index_stress_whole > 5 and index_stress_whole < 9:
                                    stress_whole.append([float(i) for i in re.findall(r'[+-]?\d+(?:\.\d+)?', line)])
                                    index_stress_whole += 1
                                # When it reaches 6+3 line,
                                elif index_stress_whole == 9:
                                    index_stress_whole = 0  # initialize index_stress_whole
                                else:
                                    index_stress_whole += 1 # Otherwise, skip a line

                            # Assign stress_individual (stress tensor on each atom) by searching "used for heat flux"
                            if re.search('used for heat flux calculation', line):
                                index_stress_individual = 1 # indicates it found that line
                            # When it found that,
                            if index_stress_individual > 0:
                                # Assign stress tensor for each atom after 4 lines until 4+NumAtoms lines
                                if index_stress_individual > 3 and index_stress_individual < (4+NumAtoms):
                                    stress_temp = [float(i) for i in re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', line)]
                                    stress_individual.append([[stress_temp[0],stress_temp[4],stress_temp[5]],[stress_temp[4],stress_temp[2],stress_temp[5]],[stress_temp[5],stress_temp[5],stress_temp[3]]])
                                    index_stress_individual += 1
                                # When it reaches 4+NumAtoms line,
                                elif index_stress_individual == 4+NumAtoms:
                                    index_stress_individual = 0  # initialize index_stres_individual
                                else:
                                    index_stress_individual += 1 # Otherwise, skip a line

                            # End of each MD step
                            if signal:
                                # Create the ASE atoms
                                atom = Atoms(
                                    numbers,
                                    positions=positions,
                                    cell=cell,
                                    pbc=pbc
                                )
                                # Assign the velocities at target temperature based on the Maxwell-Boltzmann distribution
                                MaxwellBoltzmannDistribution(atom, temperature_K=temperature, force_temp=True)

                                # Convert the format of symbols and masses for trajectory.son
                                symbols = []
                                masses = []
                                idx = 0
                                # key: atomic element, value: the total number of coressponding atoms
                                for key, value in collections.Counter(numbers_symbol).items():
                                    symbols.append([value, key])
                                    masses.append([value, mass[idx]])
                                    idx += 1

                                # Prepare the dictionary to summarize the atoms_info
                                atoms_info = {
                                    "pbc": pbc,
                                    "cell": cell,
                                    "positions": positions,
                                    "velocities": atom.get_velocities().tolist(),
                                    "symbols": symbols,
                                    "masses": masses
                                }

                                # Prepare the dictionary to summarize the calculator_info
                                calculator_info = {
                                    "energy": total_E,
                                    "forces": forces,
                                    "stress": stress_whole,
                                    "stresses": stress_individual
                                }

                                # Merge the dictionaries together
                                atom_dict = {
                                    "atoms": atoms_info,
                                    "calculator": calculator_info
                                }

                                # Initialize all indexes for next MD step
                                index_struc = 0
                                index_force = 0
                                index_stress_whole = 0
                                index_stress_individual = 0
                                signal = 0

                                # Initialize all properties for next MD step
                                cell = []
                                forces = []
                                numbers = []
                                numbers_symbol = []
                                positions = []
                                stress_whole = []
                                stress_individual = []

                                # Dump extracted atom info summary of current MD step into trajectory.son
                                son.dump(atom_dict, './../../trajectory.son', is_metadata=False)

                    # Move back to 'calc' folder
                    os.chdir(calcpath_cwd)

                else:
                    single_print(f'[harmo2son]\tCalculation has not been finished: a directory {index_calc_list[idx]}')

                    # Move back to 'calc' folder
                    os.chdir(calcpath_cwd)
            else:
                single_print(f'[harmo2son]\tCalculation has not been finished: a directory {index_calc_list[idx]}')

                # Move back to 'calc' folder
                os.chdir(calcpath_cwd)

        elif output_format == 'vibes':
            if os.path.exists(f'aims/trajectory.son'):
                metadata, data = son.load('aims/trajectory.son')
                son.dump(data[0], './../../trajectory.son', is_metadata=False)

                # Move back to 'calc' folder
                os.chdir(calcpath_cwd)
            else:
                single_print(f'[harmo2son]\tCalculation has not been finished: a directory {index_calc_list[idx]}')

                # Move back to 'calc' folder
                os.chdir(calcpath_cwd)


    single_print(f'[harmo2son]\t!! Finish converting')



def traj_run(traj_path, thermal_cutoff, num_traj, DFT_calc, num_calc):
    """Frunction [traj_run]
    Initiate FHI-aims or FHI-vibes for configurations
    from a trajectory file

    Parameters:

    traj_path: str
        Path to the trajectory file (ASE Trajectory)
    thermal_cutoff: int
        Thermalization cutoff
    num_traj: int
        The number of configurations to be calculated by DFT
    DFT_calc: str
        The name of the DFT calculator
    num_calc: int
        The number of job scripts to be submitted
    """
    from libs.lib_dft import aims_write
    import subprocess

    # Print the head
    output_init('traj_run', version, rank=0)
    single_print(f'[traj_run]\tInitiate DFT calc. for configurations from a trajectory file')

    single_print(f'[traj_run]\tRead {traj_path} file and truncate the thermalization steps {thermal_cutoff}')
    # Read the trajectory file
    traj = Trajectory(
        traj_path,
        properties='energy, forces'
        )
    # Truncate the head part until the thermalization cutoff
    traj = traj[thermal_cutoff:]
    # Collect the ramdonly picked index
    selected_traj_index = random.sample(range(len(traj)), k=num_traj)

    # Set the path to folders implementing DFT calculations
    calcpath = f'raw'
    # Create this folder
    check_mkdir(calcpath)

    # Get the current path
    mainpath_cwd = os.getcwd()

    # Move to the path to 'calc' folder implementing DFT calculations
    os.chdir(calcpath)
    # Get the new path
    calcpath_cwd = os.getcwd()

    single_print(f'[traj_run]\tGet the template of the inputs')
    # Get the template of the job script
    with open('../DFT_INPUTS/job.slurm', 'r') as job_script_DFT_initial:
        job_script_DFT_default = job_script_DFT_initial.read()
    # Prepare the command line for FHI-aims for DeepH or FHI-vibes
    if DFT_calc == 'aims':
        DFT_command = 'srun /u/kkang/programs/FHIaims-DeepH/build2/aims.220609.scalapack.mpi.x > aims.out 2>&1'
    elif DFT_calc == 'vibes':
        DFT_command = 'vibes run singlepoint aims.in &> log.aims'
    # Prepare an empty list for the calculation paths
    execute_cwd = []

    single_print(f'[traj_run]\tGo through all sampled structral configurations')
    # Go through all sampled structral configurations
    # Collect the calculations and deploy all inputs for FHI-vibes or FHI-aims
    for jndex, jtem in enumerate(selected_traj_index):
        # Get configurations until the number of target subsampling data
        if jndex < num_traj:
            # Create a folder for each structral configuration
            check_mkdir(f'{jndex}')
            # Move to that folder
            os.chdir(f'{jndex}')
            
            if DFT_calc == 'aims':
                # Check if a previous calculation exists
                if os.path.exists(f'aims.out'):
                    # Check whether calculation is finished
                    if 'Have a nice day.' in open('aims.out').read():
                        os.chdir(calcpath_cwd)
                    else:
                        # Collect the current calculation path
                        execute_cwd.append(os.getcwd())
                        # Move back to 'calc' folder
                        os.chdir(calcpath_cwd)
                else:
                    # Get FHI-aims inputs from the DFT_INPUTS folder
                    aims_write('geometry.in', traj[jtem])
                    subprocess.run(['cp', '../../DFT_INPUTS/control.in', '.'])
                    # Collect the current calculation path
                    execute_cwd.append(os.getcwd())
                    # Move back to 'calc' folder
                    os.chdir(calcpath_cwd)
            elif DFT_calc == 'vibes':
                # Check if a previous calculation exists
                if os.path.exists(f'aims/calculations/aims.out'):
                    # Check whether calculation is finished
                    if 'Have a nice day.' in open('aims/calculations/aims.out').read():
                        os.chdir(calcpath_cwd)
                    else:
                        # Collect the current calculation path
                        execute_cwd.append(os.getcwd())
                        # Move back to 'calc' folder
                        os.chdir(calcpath_cwd)
                else:
                    # Get FHI-aims inputs from the DFT_INPUTS folder
                    aims_write('geometry.in', traj[jtem])
                    subprocess.run(['cp', '../../DFT_INPUTS/aims.in', '.'])
                    # Collect the current calculation path
                    execute_cwd.append(os.getcwd())
                    # Move back to 'calc' folder
                    os.chdir(calcpath_cwd)

    single_print(f'[traj_run]\tCreate job scripts and submit them')
    # Create job scripts and submit them
    for index_calc in range(num_calc):
        job_script = f'job_{index_calc}.slurm'
        with open(job_script, 'w') as writing_input:
            writing_input.write(job_script_DFT_default)
            for index_execute_cwd, value_execute_cwd in enumerate(execute_cwd):
                if index_execute_cwd % num_calc == index_calc:
                    writing_input.write('cd '+value_execute_cwd+'\n')
                    writing_input.write(DFT_command+'\n')
        # If the previous calculation is not finished, rerun it
        subprocess.run(['sbatch', job_script])

    # Move back to the original position
    os.chdir(mainpath_cwd)
    single_print(f'[traj_run]\t!! Finish DFT calc. with the trajectory file')



def cnvg_post(nmodel, nstep):

    import matplotlib.pyplot as plt

    prd_E_data = np.load('E_matrix_prd.npz', allow_pickle=True)['E']
    prd_F_data = np.load('F_matrix_prd.npz', allow_pickle=True)['F']
    prd_S_data = np.load('S_matrix_prd.npz', allow_pickle=True)['S']

    ## Prepare the ground state structure
    # Read the ground state structure with the primitive cell
    struc_init = atoms_read('geometry.in.supercell', format='aims')
    # Get the number of atoms in unitcell
    NumAtoms = len(struc_init)

    nmodel_list = np.arange(nmodel)
    nstep_list = np.arange(nstep)
    X, Y = np.meshgrid(nmodel_list + 1, nstep_list + 1)

    prd_E_matrix_list = []
    for prd_E_step in prd_E_data:
        prd_E_matrix = np.empty([nmodel, nstep])
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                prd_E_matrix[index_nmodel, index_nstep] = prd_E_step[f'{index_nmodel}_{index_nstep}']
        prd_E_matrix_list.append(prd_E_matrix)

    prd_Eavg_matrix_list = []
    prd_Estd_matrix_list = []
    for prd_E_matrix in prd_E_matrix_list:
        prd_Eavg_matrix = np.empty([nmodel, nstep])
        prd_Estd_matrix = np.empty([nmodel, nstep])
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                prd_Eavg_matrix[index_nmodel, index_nstep] = np.average(prd_E_matrix[:(index_nmodel + 1), :(index_nstep + 1)])
                prd_Estd_matrix[index_nmodel, index_nstep] = np.std(prd_E_matrix[:(index_nmodel + 1), :(index_nstep + 1)])
        prd_Eavg_matrix_list.append(prd_Eavg_matrix)
        prd_Estd_matrix_list.append(prd_Estd_matrix)
    prd_Eavg = np.average(prd_Eavg_matrix_list, axis=0)
    prd_Estd = np.average(prd_Estd_matrix_list, axis=0)
    Z_Estd = prd_Estd / NumAtom * 1000

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, Z_Estd)
    fig.colorbar(cp)
    ax.set_title('Uncertainty of energy (meV/atom)')
    ax.set_xlabel('# of model ensemble')
    ax.set_ylabel('# of subsampling')
    plt.show()
    fig.savefig('figure_E.png')

    prd_Favg_matrix_list = []
    prd_Fstd_matrix_list = []
    for prd_F_step in prd_F_data:
        prd_Favg_matrix = np.empty([nmodel, nstep])
        prd_Fstd_matrix = np.empty([nmodel, nstep])
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                prd_F_now = []
                for idx_nmodel in range(index_nmodel + 1):
                    for idx_nstep in range(index_nstep + 1):
                        prd_F_now.append(prd_F_step[f'{idx_nmodel}_{idx_nstep}'])

                prd_F_now_avg = np.average(prd_F_now, axis=0)
                prd_F_now_norm = np.array([[np.linalg.norm(Fcomp) for Fcomp in Ftems] for Ftems in prd_F_now - prd_F_now_avg])

                prd_Favg_matrix[index_nmodel, index_nstep] = np.average(np.linalg.norm(prd_F_now_avg, axis=1))
                prd_Fstd_matrix[index_nmodel, index_nstep] = np.average(np.sqrt(np.average(prd_F_now_norm ** 2, axis=0)))
        prd_Favg_matrix_list.append(prd_Favg_matrix)
        prd_Fstd_matrix_list.append(prd_Fstd_matrix)

    prd_Favg = np.average(prd_Favg_matrix_list, axis=0)
    prd_Fstd = np.average(prd_Fstd_matrix_list, axis=0)
    Z_Fstd = prd_Fstd

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, Z_Fstd)
    fig.colorbar(cp)
    ax.set_title('Uncertainty of forces (eV/Å)')
    ax.set_xlabel('# of model ensemble')
    ax.set_ylabel('# of subsampling')
    plt.show()
    fig.savefig('figure_F.png')

    prd_S_matrix_list = []
    for prd_S_step in prd_S_data:
        prd_S_matrix = np.empty([nmodel, nstep])
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                prd_S_matrix[index_nmodel, index_nstep] = prd_S_step[f'{index_nmodel}_{index_nstep}']
        prd_S_matrix_list.append(prd_S_matrix)

    prd_Savg_matrix_list = []
    prd_Sstd_matrix_list = []
    for prd_S_matrix in prd_S_matrix_list:
        prd_Savg_matrix = np.empty([nmodel, nstep])
        prd_Sstd_matrix = np.empty([nmodel, nstep])
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                prd_Savg_matrix[index_nmodel, index_nstep] = np.average(prd_S_matrix[:(index_nmodel + 1), :(index_nstep + 1)])
                prd_Sstd_matrix[index_nmodel, index_nstep] = np.std(prd_S_matrix[:(index_nmodel + 1), :(index_nstep + 1)])
        prd_Savg_matrix_list.append(prd_Savg_matrix)
        prd_Sstd_matrix_list.append(prd_Sstd_matrix)
    prd_Savg = np.average(prd_Savg_matrix_list, axis=0)
    prd_Sstd = np.average(prd_Sstd_matrix_list, axis=0)
    Z_Sstd = prd_Sstd

    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, Y, Z_Sstd)
    fig.colorbar(cp)
    ax.set_title('Uncertainty of anharmonicity')
    ax.set_xlabel('# of model ensemble')
    ax.set_ylabel('# of subsampling')
    plt.show()
    fig.savefig('figure_S.png')



def convert_npz(name, harmonic_F):

    from libs.lib_util import get_displacements, get_fc_ha, get_E_ha

    data = np.load(name)
    E_step = []
    F_step = []

    for id_E, id_F, id_R, id_z, id_CELL, id_PBC \
    in zip(data['E'], data['F'], data['R'], data['z'], data['CELL'], data['PBC']):
        displacements = get_displacements(id_R, 'geometry.in.supercell')
        F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
        E_ha = get_E_ha(displacements, F_ha)
        if harmonic_F:
            F_step.append(np.array(id_F) - F_ha)
            E_step.append(np.array(id_E) - E_ha)
        else:
            F_step.append(np.array(id_F) + F_ha)
            E_step.append(np.array(id_E) + E_ha)

    if harmonic_F:
        name_file = name[:-4]+'_harmonic'
    else:
        name_file = name[:-4]+'_ori'

    np.savez(
        name_file,
        E=np.array(E_step),
        F=np.array(F_step),
        R=np.array(data['R']),
        z=np.array(data['z']),
        CELL=np.array(data['CELL']),
        PBC=np.array(data['PBC'])
    )