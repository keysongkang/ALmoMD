import os
import re
import sys
import subprocess
import time
import shutil
import numpy as np
import ast
from functools import lru_cache

from ase        import Atoms
from ase.data   import atomic_numbers
from ase.io.aims import read_aims as read_aims_geometry
from ase.geometry import find_mic

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

_TIMING_ENABLED = os.environ.get('ALMOMD_TIMING', '').lower() in ('1', 'true', 'yes', 'on')


def timeit(f):
    """Print wall-clock runtime of `f` when ALMOMD_TIMING=1 is set.

    Off by default so decorated hot-path functions carry zero runtime
    cost in production. Enable with e.g. `ALMOMD_TIMING=1 almomd cont`.
    Uses time.perf_counter (higher resolution than time.time) and calls
    torch.cuda.synchronize() if available so GPU work is not hidden
    behind an early return.
    """
    if not _TIMING_ENABLED:
        return f

    import functools

    try:
        import torch
        _sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    except ImportError:
        _sync = lambda: None

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        _sync()
        start = time.perf_counter()
        ret = f(*args, **kwargs)
        _sync()
        print(f"{f.__name__} runs for {time.perf_counter() - start:.3f} s",
              flush=True)
        return ret

    return wrapper


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
    os.makedirs(dir_name, exist_ok=True)
        
        
def rm_mkdir(dir_name):
    """Function [rm_mkdir]
    Check the existance of the directory.
    If it exists, remove it.

    Parameters:

    dir_name: str
        The name of the target directory
    """
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)


def rm_file(dir_name):
    """Function [rm_file]
    Check the existance of the file.
    If it exists, remove it.

    Parameters:

    dir_name: str
        The name of the target file
    """
    if os.path.exists(dir_name):
        os.remove(dir_name)


@lru_cache(maxsize=8)
def _read_aims_geometry_cached(filename):
    return read_aims_geometry(filename)


@lru_cache(maxsize=8)
def _loadtxt_cached(filename):
    return np.loadtxt(filename)


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


def get_sigma(f_data, f_model, rtol=1e-5, axis=None, silent=False):
    """
    Calculate RMSE / STD

    Args:
    ----
        f_data (np.ndarray): input data
        f_model (np.ndarray): input model data
        rtol (float): assert f.mean() / f.std() < rtol
        axis (tuple): axis along which mean and std are taken
        silent (bool): silence warnings

    """
    f1 = np.asarray(f_data)
    f2 = np.asarray(f_model)

    # check f_data_mean, should be small
    if np.any(f1.mean(axis=axis) > rtol * f1.std(axis=axis)) and not silent:
        print(f"f_data.mean(axis=axis) is {f1.mean(axis=axis)}")

    rmse = (f1 - f2).std(axis=axis)
    std = f1.std(axis=axis)

    return rmse / std

class Structure(Atoms):
    """
    A wrapper of Atoms for calculating harmonic forces and anharmonicity
    """
    def __init__(self, *args, **kwargs):
        Atoms.__init__(self, *args, **kwargs)
        self.ref_structure:Atoms = None
        self.displacements:np.ndarray = None
        self.force_constants:np.ndarray = None
        self.force_ha:np.ndarray = None
        self.e_ref = []
        self.eatoms_ref = []
        self.sigma = None

    @classmethod
    def from_atoms(cls, atoms:Atoms):
        return cls(
            numbers=atoms.numbers,
            positions=atoms.positions,
            cell=atoms.cell,
            pbc=atoms.pbc,
            velocities=atoms.get_velocities(),
        )

    def set_ref_structure(self, filename="geometry.in.supercell", *args, **kwargs):
        self.ref_structure = _read_aims_geometry_cached(filename).copy()

    def set_displacements(self, *args, **kwargs):
        if self.ref_structure is None:
            self.set_ref_structure(*args, **kwargs)
        # Get the structral information
        ref_cell = np.asarray(self.ref_structure.get_cell())
        ref_positions = np.array(self.ref_structure.get_positions())
        shape = ref_positions.shape
        step_positions = np.array(self.positions)

        # Get the displacements
        displacements = step_positions - ref_positions
        displacements = find_mic(displacements.reshape(-1, 3), ref_cell)[0]
        self.displacements = displacements.reshape(*shape)

    def get_displacements(self, *args, **kwargs) -> np.ndarray:
        if self.displacements is None:
            self.set_displacements(*args, **kwargs)
        return self.displacements

    def set_force_constants(self, filename="FORCE_CONSTANTS_remapped", *args, **kwargs):
        self.force_constants = _loadtxt_cached(filename).copy()

    def set_fc_ha(self, *args, **kwargs):
        if self.displacements is None:
            self.set_displacements()
        if self.force_constants is None:
            self.set_force_constants(*args, **kwargs)

        shape = self.displacements.shape
        fc_ha = -self.force_constants @ self.displacements.flatten()
        self.force_ha = fc_ha.reshape(shape)

    def get_fc_ha(self, *args, **kwargs):
        if self.force_ha is None:
            self.set_fc_ha(*args, **kwargs)
        return self.force_ha

    def set_E_ha(self, *args, **kwargs):
        if self.displacements is None:
            self.set_displacements(*args, **kwargs)
        if self.force_ha is None:
            self.set_fc_ha(*args, **kwargs)

        self.e_ha = self.displacements.flatten() @ -self.force_ha.flatten() / 2

    def get_E_ha(self, *args, **kwargs):
        if self.e_ha is None:
            self.set_E_ha(*args, **kwargs)
        return self.e_ha

    def set_E_ref(self, nmodel, nstep, calculator, *args, **kwargs):
        if self.ref_structure is None:
            self.set_ref_structure(*args, **kwargs)

        # # calculate in multi threading
        # GPU_threading = True
        # if GPU_threading:
        #     # initialize threading for each model
        #     t_list = []
        #     for index_nmodel in range(nmodel):
        #         for index_nstep in range(nstep):
        #             index_totalmodel = index_nmodel * nstep + index_nstep
        #             t = threading.Thread(
        #                 target=calculator[index_totalmodel].calculate,
        #                 args=[self.ref_structure,]
        #             )
        #             t_list.append(t)
        #
        #     # run each model
        #     for t in t_list:
        #         t.start()
        #
        #     # wait for another thread to finish
        #     for t in t_list:
        #         t.join()
        # else:
        #     for index_nmodel in range(nmodel):
        #         for index_nstep in range(nstep):
        #             index_totalmodel = index_nmodel * nstep + index_nstep
        #             calculator[index_totalmodel].calculate(self.ref_structure)

        # get the results
        e_ref = []
        eatom_ref = []
        zndex = 0
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                self.ref_structure.calc = calculator[zndex]
                try:
                    eatom_ref.append(np.array(self.ref_structure.get_potential_energies()))
                except Exception:
                    # print(f"error encountered in get_potential_energies: {e}")
                    eatom_ref.append(np.array([]))
                e_ref.append(self.ref_structure.get_potential_energy())
                zndex += 1

        self.e_ref = np.array(e_ref)
        self.eatom_ref = np.array(eatom_ref)

    def get_E_ref(self, *args, **kwargs):
        if not self.e_ref and not self.eatoms_ref:
            self.set_E_ref(*args, **kwargs)

        return [self.e_ref, self.eatom_ref]

    def set_sigma(self, struc_step_forces, **kwargs): # FIXME struc_step_forces
        if self.ref_structure is None:
            self.set_ref_structure()
        if self.force_constants is None:
            self.set_force_constants()
        if self.force_ha is None:
            self.set_fc_ha()

        # Get the force of the current step
        fc_step = np.array(struc_step_forces)

        # set sigma
        self.sigma = get_sigma(fc_step, self.force_ha, silent=True)

        # set sigma a
        force_a = []
        for fc_step_atom, fc_ha_atom in zip(fc_step, self.force_ha):
            force_a.append(get_sigma(fc_step_atom, fc_ha_atom, silent=True))
        self.sigma_a = force_a

    def eval_sigma(self, al_type, *args, **kwargs):
        self.set_sigma(*args, **kwargs)

        if al_type == 'sigma_max':
            return self.sigma_a
        else:
            return self.sigma

    def update_structure(self, *args, **kwargs):
        self.displacements:np.ndarray = None
        self.force_ha:np.ndarray = None
        self.e_ref = []
        self.eatoms_ref = []
        self.sigma = None


@timeit
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
    # Read the ground state structure with the primitive cell
    ref_struc_super = _read_aims_geometry_cached(struc)

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


@timeit
def get_fc_ha(displacements, fc_file='FORCE_CONSTANTS_remapped'):
    # Get the harmonic force from the force constant of the phonon dispersion
    fc = _loadtxt_cached(fc_file)
    shape = displacements.shape
    fc_ha = -fc @ displacements.flatten()

    return fc_ha.reshape(shape)

@timeit
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
            try:
                Eatom_ref.append(np.array(struc_init.get_potential_energies()))
            except Exception:
                # print(f"Error encountered in get_potential_energies: {e}")
                Eatom_ref.append(np.array([]))
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


_LITERAL_INPUTS = {
    'supercell', 'supercell_init', 'mask', 'harmonic_F', 'anharmoic_F',
    'anharmonic_F', 'meta_restart', 'signal_uncert', 'criteria_energy',
    'train_stress', 'npz_sigma', 'loss_var', 'pfactor',
}

_FLOAT_INPUTS = {
    'crtria_cnvg', 'friction', 'compressibility', 'kB', 'uncert_shift',
    'uncert_grad', 'meta_Ediff', 'meta_r_crtria', 'ttime', 'timestep',
    'cell_factor', 'bias_A', 'bias_B', 'temp_factor', 'r_cut', 'we', 'wf',
    'lr', 'lr_stop', 'lr_decay_exp_decay_factor', 'skin', 'E_gs',
}

_INT_INPUTS = {
    'ntrain_init', 'ntrain', 'nstep', 'nmodel', 'nperiod', 'temperature',
    'taut', 'pressure', 'taup', 'steps_ther', 'steps_init', 'steps_random',
    'cutoff_ther', 'lmax', 'nfeatures', 'random_index', 'wndex', 'steps',
    'loginterval', 'num_calc', 'test_index', 'num_mdl_calc', 'printinterval',
    'l', 'l_min', 'l_max', 'max_body_order', 'f_body_order', 'epochs',
    'lr_decay_exp_transition_steps', 'seed_data', 'seed_training',
}

_OPTIONAL_INT_INPUTS = {
    'size_batch', 'size_batch_training', 'size_batch_validation',
}


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

                if name in _LITERAL_INPUTS:
                    value = ast.literal_eval(value)
                elif name in _FLOAT_INPUTS:
                    value = float(value)
                elif name in _INT_INPUTS:
                    value = int(value)
                elif name in _OPTIONAL_INT_INPUTS and value != 'None':
                    value = int(value)
                else:
                    value = str(value)

                variables[name] = value

    return variables
