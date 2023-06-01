import os
import re
import sys
import son
import random
import argparse
import collections
import numpy as np
import ase.units as units
from ase import Atoms
from ase.data import atomic_numbers
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from libs.lib_util import check_mkdir, rm_file

def aims2son(temperature):
    """Function [aims2son]
    Convert aims.out to trajectory.son and assign the velocity based on the 
    Maxwell-Boltzmann distribution using the temperature.
    
    Currently it works with FHI-aims version 220506.

    Parameters:

    temperature: float
        temperature in units of Kelvin.
    """

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


def split_son(num_split, E_gs):
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


    # Read trajectory.son file
    metadata, data = son.load('trajectory.son')

    # Randomly sample testing data with a total count of num_split.
    test_data = random.sample(data, num_split)
    # Extract the training data that is not included in the testing data
    train_data = [d for d in data if d not in test_data]
    
    # Check the existance of trajectory_test.son and trajectory_train.son files,
    # because it is annoying when we mixuse these files with different sampling
    if os.path.exists('trajectory_test.son') and os.path.exists('trajectory_train.son'):
        print('The files trajectory_test.son and trajectory_train.son already exist.')
        # May need to prepare for the conversion of SON file to NPZ file
        metadata, test_data = son.load('trajectory_test.son')
    else:
        print('Collect samples for training and testing data.')
        rm_file('trajectory_test.son')
        rm_file('trajectory_train.son')
        rm_file('data/data-test.npz')
        for test_item in test_data:
            son.dump(test_item, 'trajectory_test.son')   # Save testing data into trajectory_test.son file
        for train_item in train_data:
            son.dump(train_item, 'trajectory_train.son') # Save training data into trajectory_train.son file

    ### Save testing data in form of npz file for validation error checking.
    # Create a folder named data
    check_mkdir('data')

    # Check if the data-test.npz file exists
    if os.path.exists('data/data-test.npz'):
        print('The file data-test.npz already exists.')
    else:
        # Prepare the empty lists for properties
        E_test      = [] # Total energy
        F_test      = [] # Forces
        R_test      = [] # Atomic positions
        z_test      = [] # Chemical elements
        CELL_test   = [] # Lattice paratmers
        PBC_test    = [] # Periodicity

        # Dump the informations
        for test_item in test_data:
            E_test.append(test_item['calculator']['energy'] - E_gs); # Shift the total energy by the reference value.
            F_test.append(test_item['calculator']['forces']);
            R_test.append(test_item['atoms']['positions']);
            z_test.append([atomic_numbers[item[1]] for item in test_item['atoms']['symbols'] for index in range(item[0])]); # Convert format
            CELL_test.append(test_item['atoms']['cell']);
            PBC_test.append(test_item['atoms']['pbc'])
        
        # Save all information into data-test.npz
        npz_name = 'data/data-test.npz'
        np.savez(
            npz_name[:-4],
            E=np.array(E_test),
            F=np.array(F_test),
            R=np.array(R_test),
            z=np.array(z_test),
            CELL=np.array(CELL_test),
            PBC=np.array(PBC_test)
        )
        
        print('Finish the sampling testing data: data-train.npz')


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

    # Get the template of the job script
    with open('../template/job.slurm', 'r') as job_script_DFT_initial:
        job_script_DFT_default = job_script_DFT_initial.read()
    # Prepare the command line for FHI-aims for DeepH or FHI-vibes
    if DFT_calc == 'aims':
        DFT_command = 'srun /u/kkang/programs/FHIaims-DeepH/build/aims.220609.scalapack.mpi.x > aims.out 2>&1'
    elif DFT_calc == 'vibes':
        DFT_command = 'vibes run singlepoint aims.in &> log.aims'

    # Prepare an empty list for the calculation paths
    execute_cwd = []


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
                subprocess.run(['cp', f'./../../harmonic/{harmonic_file}', 'geometry.in'])
                # Get FHI-aims inputs from the template folder
                subprocess.run(['cp', './../../template/control.in', '.'])
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
                subprocess.run(['cp', f'./../../harmonic/{harmonic_file}', 'geometry.in'])
                # Get FHI-aims inputs from the template folder
                subprocess.run(['cp', './../../template/aims.in', '.'])
                # Collect the current calculation path
                execute_cwd.append(os.getcwd())
                # Move back to 'calc' folder
                os.chdir(calcpath_cwd)

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


def harmonic2son(temperature, num_sample):
    """Frunction [harmonic_run]
    Collect all results of FHI-vibes calculations
    with harmonic samplings and convert it to SON file

    Parameters:

    temperature: float
        Temperature (K)
    num_sample: int
        The number of harmonic samples
    """

    # Prepare the index inputs
    index_temp = '{:0>4.0f}'.format(temperature)
    index_calc_list = [f'{i:03d}' for i in range(num_sample)]

    # Get the current path
    mainpath_cwd = os.getcwd()

    # Move to the calculation directory
    os.chdir(f'raw')

    # Get the full path to the calculation directotry
    calcpath_cwd = os.getcwd()

    for idx in range(num_sample):
        # Create a folder for each structral configuration
        check_mkdir(f'{idx}')
        # Move to that folder
        os.chdir(f'{idx}')

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
                print(f'Calculation has not been finished: a directory {index_calc_list[idx]}')

                # Move back to 'calc' folder
                os.chdir(calcpath_cwd)
        else:
            print(f'Calculation has not been finished: a directory {index_calc_list[idx]}')

            # Move back to 'calc' folder
            os.chdir(calcpath_cwd)



def traj_run(traj_path, thermal_cutoff, num_traj, DFT_calc):
    """Frunction [traj_run]
    Initiate FHI-aims or FHI-vibes for configurations
    from a trajectory file

    Parameters:

    traj_path: str
        Path to the trajectory file
    thermal_cutoff: int
        Thermalization cutoff
    num_traj: int
        The number of configurations to be calculated by DFT
    DFT_calc: str
        The name of the DFT calculator
    """

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

    # Get the template of the job script
    with open('../template/job.slurm', 'r') as job_script_DFT_initial:
        job_script_DFT_default = job_script_DFT_initial.read()
    # Prepare the command line for FHI-aims for DeepH or FHI-vibes
    if DFT_calc == 'aims':
        DFT_command = 'srun /u/kkang/programs/FHIaims-DeepH/build/aims.220609.scalapack.mpi.x > aims.out 2>&1'
    elif DFT_calc == 'vibes':
        DFT_command = 'vibes run singlepoint aims.in &> log.aims'
    # Prepare an empty list for the calculation paths
    execute_cwd = []

    # Go through all sampled structral configurations
    # Collect the calculations and deploy all inputs for FHI-vibes
    for jndex, jtem in enumerate(selected_traj_index):
        # Get configurations until the number of target subsampling data
        if jndex < numstep:
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
                    # Get FHI-aims inputs from the template folder
                    aims_write('geometry.in', traj[jtem])
                    subprocess.run(['cp', '../../template/control.in', '.'])
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
                    # Get FHI-aims inputs from the template folder
                    aims_write('geometry.in', traj[jtem])
                    subprocess.run(['cp', '../../template/aims.in', '.'])
                    # Collect the current calculation path
                    execute_cwd.append(os.getcwd())
                    # Move back to 'calc' folder
                    os.chdir(calcpath_cwd)

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