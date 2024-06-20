from ase.io.trajectory import Trajectory
from ase.io import write as atoms_write

import os
import subprocess
import random
import pandas as pd
import numpy as np
from decimal import Decimal

from libs.lib_util   import check_mkdir, single_print

def run_DFT(inputs):
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

    condition = f'{inputs.temperature}K-{inputs.pressure}bar'

    # Read MD trajectory file of sampled configurations
    traj_DFT = Trajectory(
        f'TRAJ/traj-{condition}_{inputs.index+1}.traj',
        properties='energy, forces'
        )
    
    if inputs.output_format == 'nequip':
        from nequip.ase import nequip_calculator
        import torch
        torch.set_default_dtype(torch.float64)

        dply_model = f'deployed-model_0_0.pth'
        if os.path.exists(f'REFER/{dply_model}'):
            single_print(f'\t\tFound the deployed model: {dply_model}')
            refer_MLIP = nequip_calculator.NequIPCalculator.from_deployed_model(
                f'REFER/{dply_model}', device=inputs.device
                )
        else:
            # If there is no model, turn on the termination signal
            single_print(f'\t\tCannot find the model: {dply_model}')
            signal = 1

    # Set the path to folders implementing DFT calculations
    calcpath = f'CALC/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index+1}'
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
    if inputs.output_format != 'nequip':
        with open(f'../../DFT_INPUTS/{inputs.job_dft_name}', 'r') as job_script_DFT_initial:
            job_script_DFT_default = job_script_DFT_initial.read()

    # Prepare an empty list for the calculation paths
    execute_cwd = []

    if inputs.calc_type == 'random':
        smapled_indices = random.sample(range(inputs.steps_random), inputs.ntotal)
    else:
        if inputs.uncert_type == 'absolute':
            uncert_piece = 'Abs'
        elif inputs.uncert_type == 'relative':
            uncert_piece = 'Rel'

        if inputs.al_type == 'energy':
            al_piece = 'E'
        elif inputs.al_type == 'force' or 'force_max':
            al_piece = 'F'
        elif inputs.al_type == 'sigma' or 'sigma_max':
            al_piece = 'S'

        data = pd.read_csv(f'./../../UNCERT/uncertainty-{condition}_{inputs.index}.txt', sep='\t')
        uncert_result = np.array(data[data['Acceptance'] == 'Accepted   ']['Uncert'+uncert_piece+'_'+al_piece])
        sorted_indices = np.argsort(uncert_result)
        smapled_indices = sorted_indices[inputs.ntotal*(-1):][::-1]

    # Go through all sampled structral configurations
    # Collect the calculations and deploy all inputs for FHI-vibes

    # if inputs.output_format == 'nequip':
    #     from ase.io import read as atoms_read
    #     single_print('[NequIP] Extract the pretrained NequIP ground truth.')
    #     struc_init = atoms_read('./../../geometry.in.supercell', format='aims')
    #     struc_init.calc = refer_MLIP
    #     E_gs = struc_init.get_potential_energy()
    #     single_print('[NequIP] Get G_gs from NequIP.')


    for jndex, jtem in enumerate(smapled_indices):
        # Get configurations until the number of target subsampling data
        if jndex < inputs.ntotal:
            # Create a folder for each structral configuration
            check_mkdir(f'{jndex}')
            # Move to that folder
            os.chdir(f'{jndex}')
        
            if inputs.output_format == 'nequip':
                from ase.io.trajectory import TrajectoryWriter
                write_geo = TrajectoryWriter('geometry.traj', mode='w')
                refer_atom = traj_DFT[jtem]
                refer_atom.calc = refer_MLIP

                # refer_E = refer_atom.get_potential_energy() - E_gs
                refer_E = refer_atom.get_potential_energy()
                refer_F = refer_atom.get_forces()
                write_geo.write(refer_atom)
                write_geo.close()
                os.chdir(calcpath_cwd)
            else:
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
    if inputs.output_format != 'nequip':
        for index_calc in range(inputs.num_calc):
            job_script = f'{inputs.job_dft_name.split(".")[0]}_{index_calc}.{inputs.job_dft_name.split(".")[1]}'
            with open(job_script, 'w') as writing_input:
                writing_input.write(job_script_DFT_default)
                for index_execute_cwd, value_execute_cwd in enumerate(execute_cwd):
                    if index_execute_cwd % inputs.num_calc == index_calc:
                        writing_input.write('cd '+value_execute_cwd+'\n')
                        writing_input.write(inputs.vibes_command+'\n')
            # If the previous calculation is not finished, rerun it
            # subprocess.run([inputs.job_command, job_script])
            # os.system(f'{inputs.job_command} {job_script}')

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