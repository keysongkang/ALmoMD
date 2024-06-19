import os
import re
import sys
import subprocess
import numpy as np

from ase        import Atoms
from ase.data   import atomic_numbers


# def mpi_print(string, rank):
#     """Function [mpi_print]
#     Instantly print a message only from one MPI thread.

#     Parameters:

#     string: str
#         A message to be printed
#     rank: int
#         The index of current MPI thread
#     """

#     if rank == 0:
#         print(string)
#     sys.stdout.flush() # Instant printing


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


def output_init(string, version):
    from datetime import datetime

    if not string == 'cont' and not string == 'gen':
        single_print(
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
            '#################################################################\n'
        )

    single_print(f'[{string}]\t' + datetime.now().strftime("Date/Time: %Y %m %d %H:%M"))
    single_print(f'[{string}]\tALmoMD Version: {version}')


        
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


def eval_sigma(struc_step_forces, struc_step_positions, al_type):
    """Function [read_input_file]
    Read 'input.in' and assign variables.

    Parameters:

    file_path: str
        File path of the file

    Returns:

    variables: dictionary
        A dictionary containing all new variables
    """

    from vibes.anharmonicity_score import get_sigma

    displacements = get_displacements(struc_step_positions, 'geometry.in.supercell')
    fc_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')

    # Get the force of the current step
    fc_step = np.array(struc_step_forces)

    if al_type == 'sigma_max':
        force_a = []
        for fc_step_atom, fc_ha_atom in zip(fc_step, fc_ha):
            force_a.append(get_sigma(fc_step_atom, fc_ha_atom, silent=True))
        return force_a
    else:
        return get_sigma(fc_step, fc_ha, silent=True)



def get_displacements(struc_step_positions, struc='geometry.in.supercell'):

    from ase.geometry import find_mic
    from ase.io.aims import read_aims

    # Read the ground state structure with the primitive cell
    ref_struc_super = read_aims(struc)

    # Get the structral information
    ref_cell = np.asarray(ref_struc_super.get_cell())
    ref_positions = np.array(ref_struc_super.get_positions())
    shape = ref_positions.shape
    step_positions = np.array(struc_step_positions)

    # Get the displacements
    displacements = step_positions - ref_positions
    displacements = find_mic(displacements.reshape(-1, 3), ref_cell)[0]
    displacements = displacements.reshape(*shape)

    return displacements



def get_fc_ha(displacements, fc_file='FORCE_CONSTANTS_remapped'):
    # Get the harmonic force from the force constant of the phonon dispersion
    fc = np.loadtxt(fc_file)
    shape = displacements.shape
    fc_ha = -fc @ displacements.flatten()

    return fc_ha.reshape(shape)


def get_E_ha(displacements, fc_ha):
    return displacements.flatten() @ -fc_ha.flatten() / 2


def get_E_ref(nmodel, nstep, calculator):
    # Read the ground state structure with the primitive cell
    from ase.io import read as atoms_read
    struc_init = atoms_read('geometry.in.supercell', format='aims')

    E_ref = []
    Eatom_ref = []
    zndex = 0
    for index_nmodel in range(nmodel):
        for index_nstep in range(nstep):
            struc_init.calc = calculator[zndex]
            Eatom_ref.append(np.array(struc_init.get_potential_energies()))
            E_ref.append(struc_init.get_potential_energy())
            zndex += 1

    return [np.array(E_ref), np.array(Eatom_ref)]


def generate_msg(al_type):
    result_msg = 'Temperature[K]\tIteration\t'\
                 + 'TestError_E\tTestError_F\tTestError_S\t'\
                 + 'E_potent_avg_i\tE_potent_std_i'

    if al_type == 'energy' or al_type == 'energy_max':
        result_msg += '\tUn_Abs_E_avg_i\tUn_Abs_E_std_i'\
                      + '\tUn_Rel_E_avg_i\tUn_Rel_E_std_i'

    if al_type == 'energy_max':
        result_msg += '\tUn_Abs_Ea_avg_i\tUn_Abs_Ea_std_i'\
                      + '\tUn_Rel_Ea_avg_i\tUn_Rel_Ea_std_i'

    if al_type == 'force' or al_type == 'force_max':
        result_msg += '\tUn_Abs_F_avg_i\tUn_Abs_F_std_i'\
                      + '\tUn_Rel_F_avg_i\tUn_Rel_F_std_i'

    if al_type == 'sigma' or al_type == 'sigma_max':
        result_msg += '\tUn_Abs_S_avg_i\tUn_Abs_S_std_i'\
                      + '\tUn_Rel_S_avg_i\tUn_Rel_S_std_i'

    if al_type == 'energy' or al_type == 'energy_max':
        result_msg += '\tUn_Abs_E_avg_a\tUn_Rel_E_avg_a'

    if al_type == 'force' or al_type == 'force_max':
        result_msg += '\tUn_Abs_F_avg_a\tUn_Rel_F_avg_a'

    if al_type == 'sigma' or al_type == 'sigma_max':
        result_msg += '\tUn_Abs_S_avg_a\tUn_Rel_S_avg_a'

    return result_msg


class empty_inputs:
    pass


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
                value = value.split('#')[0].strip()

                # Perform type conversions for specific variables
                if name in [
                'supercell', 'supercell_init', 'mask', 'harmonic_F', 'anharmoic_F', 'meta_restart',
                'signal_uncert', 'criteria_energy', 'train_stress', 'npz_sigma', 'E_gs', 'lr', 'lr_stop'
                ]:
                    value = eval(value)
                elif name in [
                'crtria_cnvg', 'friction', 'compressibility', 'kB', 'uncert_shift', 'uncert_grad',
                'meta_Ediff', 'meta_r_crtria', 'ttime', 'pfactor', 'timestep', 'cell_factor', 'bias_A',
                'bias_B', 'temp_factor', 'r_cut', 'we', 'wf', 'ws', 'eval_energy_t',
                'lr_decay_exp_decay_factor', 'clip_by_global_norm'
                ]:
                    value = float(value)
                elif name in [
                'ntrain_init', 'ntrain', 'nstep', 'nmodel', 'nperiod', 'temperature', 'taut', 'pressure',
                'taup', 'steps_ther', 'steps_init', 'steps_random', 'cutoff_ther', 'lmax', 'nfeatures',
                'random_index', 'wndex', 'steps', 'loginterval', 'num_calc', 'test_index', 'num_mdl_calc',
                'printinterval', 'idx_atom', 'l', 'f', 'l_min', 'l_max', 'max_body_order', 'f_body_order',
                'epochs', 'lr_decay_exp_transition_steps', 'size_batch', 'size_batch_training',
                'size_batch_validation', 'seed_data', 'seed_training', 'skin'
                ]:
                    value = int(value)
                else:
                    value = str(value)

                variables[name] = value

    return variables
