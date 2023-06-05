import os
import re
import sys
import subprocess
import numpy as np

from ase        import Atoms
from ase.data   import atomic_numbers


def mpi_print(string, rank):
    """Function [mpi_print]
    Instantly print a message only from one MPI thread.

    Parameters:

    string: str
        A message to be printed
    rank: int
        The index of current MPI thread
    """

    if rank == 0:
        print(string)
    sys.stdout.flush() # Instant printing


def single_print(string):
    """Function [single_print]
    Instantly print a message (Use it a function with one thread)

    Parameters:

    string: str
        A message to be printed
    """
    print(string)
    sys.stdout.flush() # Instant printing

        
def check_mkdir(dir_name):
    """Function [check_mkdir]
    Check the existance of the directory.
    If it does not exist, create it.

    Parameters:

    dir_name: str
        The name of the new directory
    """
    if os.path.exists('./'+dir_name) == False:
        os.system('mkdir ./'+dir_name)
        
        
def rm_mkdir(dir_name):
    """Function [rm_mkdir]
    Check the existance of the directory.
    If it exists, remove it.

    Parameters:

    dir_name: str
        The name of the target directory
    """
    if os.path.exists('./'+dir_name):
        os.system('rm -r ./'+dir_name)


def rm_file(dir_name):
    """Function [rm_file]
    Check the existance of the file.
    If it exists, remove it.

    Parameters:

    dir_name: str
        The name of the target file
    """
    if os.path.exists('./'+dir_name):
        os.system('rm ./'+dir_name)


def output_init(string, version, MPI=False):
    from datetime import datetime

    if MPI:
        from mpi4py import MPI
        # Extract MPI infos
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    else:
        rank = 0

    mpi_print(
        '\n'
        '#################################################################\n'
        '#                                                               #\n'
        '#     @       @                             @@   @@   @@@@@     #\n'
        '#     @       @                             @ @ @ @   @    @    #\n'
        '#    @ @      @                             @ @ @ @   @     @   #\n'
        '#    @ @      @                             @  @  @   @     @   #\n'
        '#   @   @     @        @ @@ @@     @@@@@    @     @   @     @   #\n'
        '#   @@@@@     @         @  @  @   @     @   @     @   @     @   #\n'
        '#  @     @    @         @  @  @   @     @   @     @   @    @    #\n'
        '#  @     @    @@@@@@@   @  @  @    @@@@@    @     @   @@@@@     #\n'
        '#                                                               #\n'
        '#################################################################\n',
        rank
        )

    mpi_print(f'[{string}]\t' + datetime.now().strftime("Date/Time: %Y %m %d %H:%M"), rank)

    mpi_print(f'[{string}]\tALmoMD Version: {version}', rank)

        
        
def job_dependency(job_str, num_jobs):
    """Function [job_dependency]
    Since it is necessaryt to wait for calculations for DFT or NequIP,
    this script submit the next job script with the dependency.
    The job dependency is based on the Slurm system.

    Parameters:

    job_str: str
        Option for different job_scripts; job-cont or job-gen
    num_jobs: int
        The number of previous job scripts
    """

    # Check output file
    item_index = 'almomd.out'
    
    # Filter out all previously submitted job IDs
    bashcommand = f'grep Submitted {item_index}'
    result = subprocess.Popen(bashcommand.split(), stdout=subprocess.PIPE)
    output, error = result.communicate()
    
    # Collect all previously submitted job IDs
    job_index = re.findall("\d+", output.decode("utf-8"))
    dependency_index = ''
    for jtem in job_index[(-1)*num_jobs:]:
        dependency_index += f'{jtem},'
    
    # Submit a job dependency
    if job_str == 'cont':
        subprocess.run(['sbatch', f'--dependency=afterany:{dependency_index[:-1]}', 'job-cont.slurm'])
    elif job_str == 'gen':
        subprocess.run(['sbatch', f'--dependency=afterany:{dependency_index[:-1]}', 'job-gen.slurm'])
    else:
        single_print('Initialization is finished. You need to assign the calculation.')
        
        
def read_aims(file_name):
    """Function [read_aims]
    Get atomic structure, total energy, and forces from FHI-aims output.
    It is compatible with the FHI-aims 220506 version.

    Parameters:

    file_name: str
        File name of FHI-atims output

    Returns:

    atom: ASE Atoms
        Atomic struture from FHI-aims output
    total_E: float
        Total energy from FHI-aims output
    forces: np.array of float
        Forces from FHI-atims output
    """

    # Initialization of index
    index_struc = 0
    index_force = 0

    # Prepare the empty lists for quantities
    cell = []
    forces = []
    numbers = []
    positions = []
    pbc = [True, True, True]
    NumAtoms = 0

    with open(file_name,"r") as file_one: # Open the output file
        for line in file_one: # Go through line by line
            # Get the number of atoms in the simulation cell
            if re.search('Number of atoms', line):
                NumAtoms = int(re.findall(r'\d+', line)[0])

            # Get the total energy
            if re.search('Total energy corrected', line):
                total_E = float(re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', line)[0])

            # Get lattice parameters and atomic positions
            if re.search('Atomic structure that was used in the preceding time step of the wrapper', line):
                index_struc = 1
            if index_struc > 0:
                if index_struc > 2 and index_struc < 6:
                    cell.append([float(i) for i in re.findall(r'[+-]?\d+(?:\.\d+)?', line)])
                    index_struc += 1
                elif index_struc > 6 and index_struc < (7+NumAtoms):
                    positions.append([float(i) for i in re.findall(r'[+-]?\d+(?:\.\d+)?', line)])
                    numbers.append(atomic_numbers[(line[-3:].replace(' ', '')).replace('\n','')])
                    index_struc += 1
                elif index_struc == (7+NumAtoms):
                    index_struc = 0
                else:
                    index_struc += 1

            # Get forces
            if re.search('Total atomic forces', line):
                index_force = 1
            if index_force > 0:
                if index_force > 1 and index_force < (2+NumAtoms):
                    forces.append([float(i) for i in re.findall(r'[+\-]?(?:0|[1-9]\d*)(?:\.\d*)?(?:[eE][+\-]?\d+)', line)])
                    index_force += 1
                elif index_force == (2+NumAtoms):
                    index_force = 0
                else:
                    index_force += 1

    # Save structral information in a format of ASE Atoms
    atom = Atoms(
        numbers,
        positions = positions,
        cell = cell,
        pbc = pbc
    )
    
    return atom, total_E, np.array(forces)


def read_input_file(file_path):
    """Function [read_input_file]
    Read 'input.in' and assign variables.

    Parameters:

    file_path: str
        File path of the file

    Returns:

    variables: dictionary
        A dictionary containing all new variables
    """

    # Prepare an empty dictionary
    variables = {}

    with open(file_path) as f: # Open a target file
        for line in f: # Go through line by line
            line = line.strip()
            if line and not line.startswith('#'): # Skip lines starting with a # symbol
                if ':' in line: # Detect a : symbol
                    name, value = line.split(':', 1)
                elif '=' in line: # Detect a = symbol
                    name, value = line.split('=', 1)
                else:
                    continue

                name = name.strip()
                value = value.strip()

                # Perform type conversions for specific variables
                if name in ['supercell', 'supercell_init', 'mask']:
                    value = eval(value)
                elif name in ['crtria_cnvg', 'friction', 'compressibility', 'kB', 'E_gs']:
                    value = float(value)
                elif name in [
                'ntrain_init', 'ntrain', 'nstep', 'nmodel', 'temperature', 'taut',
                'pressure', 'taup', 'steps_ther', 'steps_init', 'steps_random',
                'timestep', 'cutoff_ther', 'lmax', 'nfeatures', 'random_index',
                'wndex', 'steps', 'loginterval', 'num_calc', 'test_index'
                ]:
                    value = int(value)
                else:
                    value = str(value)

                variables[name] = value

    return variables
