from ase.io.trajectory import Trajectory
from ase.io import write as atoms_write

import os
import subprocess
import random
import pandas as pd
import numpy as np
from decimal import Decimal

from libs.lib_util   import check_mkdir


def run_DFT(temperature, pressure, index, numstep, num_calc, uncert_type, al_type):
    """Function [get_criteria_uncert]
    Create a folder and run DFT calculations
    for sampled structral configurations

    Parameters:

    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    index: int
        The index of AL interactive step
    numstep: int
        The number of all sampled configurations
    num_calc: int
        The number of job scripts to be submitted
    """

    # Read MD trajectory file of sampled configurations
    traj_DFT = Trajectory(
        f'TRAJ/traj-{temperature}K-{pressure}bar_{index+1}.traj',
        properties='energy, forces'
        )
    
    # Set the path to folders implementing DFT calculations
    calcpath = f'CALC/{temperature}K-{pressure}bar_{index+1}'
    # Create these folders
    check_mkdir(f'CALC')
    check_mkdir(calcpath)

    # Get the current path
    mainpath_cwd = os.getcwd()

    # Move to the path to 'calc' folder implementing DFT calculations
    os.chdir(calcpath)
    # Get the new path
    calcpath_cwd = os.getcwd()

    # Get the template of the job script
    with open('../../DFT_INPUTS/job-vibes.slurm', 'r') as job_script_DFT_initial:
        job_script_DFT_default = job_script_DFT_initial.read()
    # Prepare the command line for FHI-vibes
    vibes_command = 'vibes run singlepoint aims.in &> log.aims'
    # Prepare an empty list for the calculation paths
    execute_cwd = []

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

    data = pd.read_csv(f'./../../UNCERT/uncertainty-{temperature}K-{pressure}bar_{index}.txt', sep='\t')
    uncert_result = np.array(data[data['Acceptance'] == 'Accepted   ']['Uncert'+uncert_piece+'_'+al_piece])
    sorted_indices = np.argsort(uncert_result)
    smapled_indices = sorted_indices[numstep*(-1):][::-1]

    # Go through all sampled structral configurations
    # Collect the calculations and deploy all inputs for FHI-vibes
    for jndex, jtem in enumerate(smapled_indices):
        # Get configurations until the number of target subsampling data
        if jndex < numstep:
            # Create a folder for each structral configuration
            check_mkdir(f'{jndex}')
            # Move to that folder
            os.chdir(f'{jndex}')
        
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
                aims_write('geometry.in', traj_DFT[jtem])
                subprocess.run(['cp', '../../../DFT_INPUTS/aims.in', '.'])
                # Collect the current calculation path
                execute_cwd.append(os.getcwd())
                # Move back to 'calc' folder
                os.chdir(calcpath_cwd)
    
    # Create job scripts and submit them
    for index_calc in range(num_calc):
        job_script = f'job-vibes_{index_calc}.slurm'
        with open(job_script, 'w') as writing_input:
            writing_input.write(job_script_DFT_default)
            for index_execute_cwd, value_execute_cwd in enumerate(execute_cwd):
                if index_execute_cwd % num_calc == index_calc:
                    writing_input.write('cd '+value_execute_cwd+'\n')
                    writing_input.write(vibes_command+'\n')
        # If the previous calculation is not finished, rerun it
        # subprocess.run(['sbatch', job_script])
        # os.system(f'sbatch {job_script}')

    # Move back to the original position
    os.chdir(mainpath_cwd)
    
    
    
def aims_write(filename, atoms):
    """Function [aims_write]
    Write FHI-aims input 'geometry.in' using atomic position and velocities

    Parameters:

    filename: str
        The name of an input file
    atoms: ASE atoms
        Sampled structural configuration
    """

    # There is a ratio difference of velocities
    # between trajectory.son and geometry.in
    velo_unit_conv = 98.22694788
    trajfile = open(filename, 'w')

    # Write lattice parameters
    for jndex in range(3):
        trajfile.write(
            f'lattice_vector ' +
            '{:.14f}'.format(Decimal(str(atoms.get_cell()[jndex,0]))) +
            ' ' +
            '{:.14f}'.format(Decimal(str(atoms.get_cell()[jndex,1]))) +
            ' ' +
            '{:.14f}'.format(Decimal(str(atoms.get_cell()[jndex,2]))) +
            '\n'
        )

    # Write atomic positions with velocities
    for kndex in range(len(atoms)):
        trajfile.write(
            f'atom ' +
            '{:.14f}'.format(Decimal(str(atoms.get_positions()[kndex,0]))) +
            ' ' +
            '{:.14f}'.format(Decimal(str(atoms.get_positions()[kndex,1]))) +
            ' ' +
            '{:.14f}'.format(Decimal(str(atoms.get_positions()[kndex,2]))) +
            ' ' +
            atoms.get_chemical_symbols()[kndex] +
            '\n'
        )
        # trajfile.write(
        #     f'    velocity ' +
        #     '{:.14f}'.format(Decimal(str(atoms.get_velocities()[kndex,0]*velo_unit_conv))) +
        #     ' ' +
        #     '{:.14f}'.format(Decimal(str(atoms.get_velocities()[kndex,1]*velo_unit_conv))) +
        #     ' ' +
        #     '{:.14f}'.format(Decimal(str(atoms.get_velocities()[kndex,2]*velo_unit_conv))) +
        #     '\n'
        # )
    trajfile.close()