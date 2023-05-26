from ase.io.trajectory import Trajectory
from ase.io import write as atoms_write

import os
import subprocess
from decimal import Decimal

from libs.lib_util   import check_mkdir


def run_DFT(temperature, pressure, index, numstep):
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
    """

    # Read MD trajectory file of sampled configurations
    traj_DFT = Trajectory(
        f'traj-{temperature}K-{pressure}bar_{index+1}.traj',
        properties='energy, forces'
        )
    
    # Set the path to folders implementing DFT calculations
    calcpath = f'calc/{temperature}K-{pressure}bar_{index+1}'
    # Create these folders
    check_mkdir(f'calc')
    check_mkdir(calcpath)

    # Get the current path
    mainpath_cwd = os.getcwd()

    # Move to the path to 'calc' folder implementing DFT calculations
    os.chdir(calcpath)
    # Get the new path
    calcpath_cwd = os.getcwd()
    
    # Go through all sampled structral configurations
    for jndex, jtem in enumerate(traj_DFT):
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
                    # If the previous calculation is not finished, rerun it
                    subprocess.run(['sbatch', 'job-vibes.slurm'])
                    # Move back to 'calc' folder
                    os.chdir(calcpath_cwd)
            else:
                # Get FHI-aims inputs from the template folder and run DFT
                aims_write('geometry.in', jtem)
                subprocess.run(['cp', '../../../template/aims.in', '.'])
                subprocess.run(['cp', '../../../template/job-vibes.slurm', '.'])
                subprocess.run(['sbatch', 'job-vibes.slurm'])
                # Move back to 'calc' folder
                os.chdir(calcpath_cwd)
    
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
        trajfile.write(
            f'    velocity ' +
            '{:.14f}'.format(Decimal(str(atoms.get_velocities()[kndex,0]*velo_unit_conv))) +
            ' ' +
            '{:.14f}'.format(Decimal(str(atoms.get_velocities()[kndex,1]*velo_unit_conv))) +
            ' ' +
            '{:.14f}'.format(Decimal(str(atoms.get_velocities()[kndex,2]*velo_unit_conv))) +
            '\n'
        )
    trajfile.close()