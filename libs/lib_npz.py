import os
import random
import numpy as np
from tqdm import tqdm
from ase.data   import atomic_numbers
from ase.io     import read as atoms_read

from libs.lib_util   import read_aims, single_print
import son



def generate_npz_DFT_init(
    traj, ntrain, nval, nstep, E_gs, workpath, harmonic_F
):
    """Function [generate_npz_DFT_init]
    Generate the initial training data sets from trajectory

    Parameters:

    traj: ASE trajectory
        Training data will be randomly sampled from this trajectory
    ntrain: int
        The total number of training data
    nval: int
        The total number of validating data
    nstep: int
        The number of subsampling sets
    E_gs: float
        Reference total energy in units of eV/Unitcell
        to shift the total energies of the trajectory
        and avoid unusually high total energy values
        that may lead to unusual weightings with force values.
        Recommend to use the ground state total energy.
    workpath: str
        The path to the working directory
    """

    # Prepare the empty list of propreties for each subsampling set
    E_train      = [[] for i in range(nstep)];
    F_train      = [[] for i in range(nstep)];
    R_train      = [[] for i in range(nstep)];
    z_train      = [[] for i in range(nstep)];
    CELL_train   = [[] for i in range(nstep)];
    PBC_train    = [[] for i in range(nstep)];
    
    # The total number of data we need for the training process
    total_ntrain = (ntrain+nval) * nstep

    single_print(f'[npz]\tSample {nstep} different training data\n')
    # Random sampling for the structural configurations from trajectory
    for i, step in zip(
        random.sample(range(0,len(traj)),total_ntrain),
        tqdm(range(total_ntrain))
        ):
        for index_nstep in range(nstep):
            if step < (ntrain+nval) * (index_nstep + 1) and step >= (ntrain+nval) * (index_nstep):
                # Energy is shifted by the reference energy
                # to avoid the unsual weighting with forces in NequIP

                if harmonic_F:
                    from libs.lib_util import get_displacements, get_fc_ha, get_E_ha
                    displacements = get_displacements(traj[i]['atoms']['positions'], 'geometry.in.supercell')
                    F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
                    F_step = np.array(traj[i]['calculator']['forces']) - F_ha
                    E_ha = get_E_ha(displacements, F_ha)
                    E_step = np.array(traj[i]['calculator']['energy']) - E_gs - E_ha
                else:
                    F_step = np.array(traj[i]['calculator']['forces'])
                    E_step = np.array(traj[i]['calculator']['energy']) - E_gs

                E_train[index_nstep].append(E_step)
                F_train[index_nstep].append(F_step);
                R_train[index_nstep].append(traj[i]['atoms']['positions']);
                z_train[index_nstep].append([atomic_numbers[item[1]] for item in traj[i]['atoms']['symbols'] for jdx in range(item[0])]);
                CELL_train[index_nstep].append(traj[i]['atoms']['cell']);
                PBC_train[index_nstep].append(traj[i]['atoms']['pbc'])
                break

    # Split the sampled data into individual files for each subsampling set
    for index_nstep in range(nstep):
        E_train_store    = E_train[index_nstep]
        F_train_store    = F_train[index_nstep]
        R_train_store    = R_train[index_nstep]
        z_train_store    = z_train[index_nstep]
        CELL_train_store = CELL_train[index_nstep]
        PBC_train_store  = PBC_train[index_nstep]

        # Save each subsampling data
        npz_name = f'{workpath}/data-train_{index_nstep}.npz'
        np.savez(
            npz_name[:-4],
            E = np.array(E_train_store),
            F = np.array(F_train_store),
            R = np.array(R_train_store),
            z = np.array(z_train_store),
            CELL = np.array(CELL_train_store),
            PBC = np.array(PBC_train_store)
            )

    single_print('[npz]\tFinish the sampling process: data-train_*.npz')
    
    
def generate_npz_DFT(
    ntrain, nval, nstep, E_gs, index,
    temperature, output_format, pressure, workpath, harmonic_F
):
    """Function [generate_npz_DFT]
    Generate training data sets
    by adding new data from the trajectory file (DFT results)
    to previous training data sets

    Parameters:

    ntrain: int
        The total number of training data
    nval: int
        The total number of validating data
    nstep: int
        The number of subsampling sets
    E_gs: float
        Reference total energy in units of eV/Unitcell
        to shift the total energies of the trajectory
        and avoid unusually high total energy values
        that may lead to unusual weightings with force values.
        Recommend to use the ground state total energy.
    index: int
        The index of AL interactive step

    temperature: float
        The desired temperature in units of Kelvin (K)
    output_format: str
        The format of output file that be read to add new data
        'aims.out' or 'trajectory.son'
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    workpath: str
        The path to the working directory
    """

    # Prepare the empty list of propreties for each subsampling set
    E_train      = [[] for i in range(nstep)];
    F_train      = [[] for i in range(nstep)];
    R_train      = [[] for i in range(nstep)];
    z_train      = [[] for i in range(nstep)];
    CELL_train   = [[] for i in range(nstep)];
    PBC_train    = [[] for i in range(nstep)];
    
    # The total number of data we need for the training process
    total_ntrain = (ntrain+nval) * nstep

    # Check the existence of NPZ files containing previous data
    npz_check = [os.path.exists(f'{workpath}/data-train_{index_nstep}.npz')\
                 for index_nstep in range(nstep)];

    # For first run, of course there is no NPZ file
    if all(npz_check) == False: # If there is any missing NPZ file,
        del npz_check
        single_print(f'[npz]\tSample {nstep} different training data\n')

        # Randomly sample the new data
        for i, step in zip(
            random.sample(range(0,total_ntrain),total_ntrain),
            tqdm(range(total_ntrain))
            ):
            # Collect these new data for each subsampling
            for index_nstep in range(nstep):
                if step < (ntrain+nval) * (index_nstep + 1) and step >= (ntrain+nval) * (index_nstep):
                    if output_format == 'aims.out':
                        # Convert 'aims.out' format to ASE trajectory format
                        atoms, atoms_potE, atoms_forces = read_aims(
                            f'./CALC/{temperature}K-{pressure}bar_{index}/{i}/aims/calculations/aims.out'
                            )
                        # Energy is shifted by the reference energy
                        # to avoid the unsual weighting with forces in NequIP

                        if harmonic_F:
                            from libs.lib_util import get_displacements, get_fc_ha, get_E_ha
                            displacements = get_displacements(atoms.get_positions(), 'geometry.in.supercell')
                            F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
                            F_step = np.array(atoms_forces) - F_ha
                            E_ha = get_E_ha(displacements, F_ha)
                            E_step = atoms_potE - E_gs - E_ha
                        else:
                            F_step = np.array(atoms_forces)
                            E_step = atoms_potE - E_gs

                        E_train[index_nstep].append(E_step);
                        F_train[index_nstep].append(F_step);
                        R_train[index_nstep].append(atoms.get_positions());
                        z_train[index_nstep].append(atoms.numbers);
                        CELL_train[index_nstep].append(atoms.get_cell());
                        PBC_train[index_nstep].append(atoms.get_pbc());
                        break
                    elif output_format == 'trajectory.son':
                        # Convert 'trajectory.son' format to ASE trajectory format
                        metadata, data = son.load(
                            f'./CALC/{temperature}K-{pressure}bar_{index}/{i}/aims/trajectory.son'
                            )
                        atom_numbers = []
                        for items in data[0]['atoms']['symbols']:
                            for jndex in range(items[0]):
                                atom_numbers.append(atomic_numbers[items[1]])
                        # Energy is shifted by the reference energy
                        # to avoid the unsual weighting with forces in NequIP

                        if harmonic_F:
                            from libs.lib_util import get_displacements, get_fc_ha, get_E_ha
                            displacements = get_displacements(data[0]['atoms']['positions'], 'geometry.in.supercell')
                            F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
                            F_step = np.array(data[0]['calculator']['forces']) - F_ha
                            E_ha = get_E_ha(displacements, F_ha)
                            E_step = np.array(data[0]['calculator']['energy']) - E_gs - E_ha
                        else:
                            F_step = np.array(data[0]['calculator']['forces'])
                            E_step = np.array(data[0]['calculator']['energy']) - E_gs

                        E_train[index_nstep].append(E_step);
                        F_train[index_nstep].append(F_step);
                        R_train[index_nstep].append(data[0]['atoms']['positions']);
                        z_train[index_nstep].append(np.array(atom_numbers));
                        CELL_train[index_nstep].append(data[0]['atoms']['cell']);
                        PBC_train[index_nstep].append(data[0]['atoms']['pbc']);
                        break
                    else:
                        single_print('[npz]\tYou need to define the output format.')

        # Merge new data with previous data
        # Split the sampled data into individual files for each subsampling set
        for index_nstep in range(nstep):
            # Path to previous data
            npz_previous = f'./MODEL/{temperature}K-{pressure}bar_{index-1}'\
                           + f'/data-train_{index_nstep}.npz'
            
            # When there is previous data, merge them together
            if os.path.exists(npz_previous):
                data_train       = np.load(npz_previous)
                E_train_store    = np.concatenate\
                ((data_train['E'], E_train[index_nstep]), axis=0)
                F_train_store    = np.concatenate\
                ((data_train['F'], F_train[index_nstep]), axis=0)
                R_train_store    = np.concatenate\
                ((data_train['R'], R_train[index_nstep]), axis=0)
                z_train_store    = np.concatenate\
                ((data_train['z'], z_train[index_nstep]), axis=0)
                CELL_train_store = np.concatenate\
                ((data_train['CELL'], CELL_train[index_nstep]), axis=0)
                PBC_train_store  = np.concatenate\
                ((data_train['PBC'], PBC_train[index_nstep]), axis=0)
            else: ##!! I don't think this part is no longer needed
                E_train_store    = E_train[index_nstep]
                F_train_store    = F_train[index_nstep]
                R_train_store    = R_train[index_nstep]
                z_train_store    = z_train[index_nstep]
                CELL_train_store = CELL_train[index_nstep]
                PBC_train_store  = PBC_train[index_nstep]

            # Path to new data
            npz_name = f'{workpath}/data-train_{index_nstep}.npz'
            # Save each subsampling data
            np.savez(
                npz_name[:-4],\
                E=np.array(E_train_store),\
                F=np.array(F_train_store),\
                R=np.array(R_train_store),\
                z=np.array(z_train_store),\
                CELL=np.array(CELL_train_store),\
                PBC=np.array(PBC_train_store)
            )
        single_print('[npz]\tFinish the sampling process: data-train_*.npz')
    else:
        single_print('[npz]\tFound all sampled training data: data-train_*.npz')

        
        
def generate_npz_DFT_rand_init(
    traj, ntrain, nval, nstep, E_gs, workpath, harmonic_F
):
    """Function [generate_npz_DFT_rand_init]
    Generate the initial training data sets from trajectory.
    The main distinction from the function [generate_npz_DFT_init] is
    that this function returnsa list of randomly selected sample indices
    (traj_idx).

    Parameters:

    traj: ASE trajectory
        Training data will be randomly sampled from this trajectory
    ntrain: int
        The total number of training data
    nval: int
        The total number of validating data
    nstep: int
        The number of subsampling sets
    E_gs: float
        Reference total energy in units of eV/Unitcell
        to shift the total energies of the trajectory
        and avoid unusually high total energy values
        that may lead to unusual weightings with force values.
        Recommend to use the ground state total energy.
    workpath: str
        The path to the working directory ##!! this or temperature/pressrue/index might be removed

    Returns:
    traj_idx: list of int
        The list of randomly selected sample indices
    """

    # Prepare the empty list of propreties for each subsampling set
    E_train      = [[] for i in range(nstep)];
    F_train      = [[] for i in range(nstep)];
    R_train      = [[] for i in range(nstep)];
    z_train      = [[] for i in range(nstep)];
    CELL_train   = [[] for i in range(nstep)];
    PBC_train    = [[] for i in range(nstep)];
    traj_idx     = [];
    
    # The total number of data we need for the training process
    total_ntrain = (ntrain+nval) * nstep

    single_print(f'[npz]\tSample {nstep} different training data\n')
    # Random sampling for the structural configurations from trajectory
    for i, step in zip(
        random.sample(range(0,len(traj)),total_ntrain),
        tqdm(range(total_ntrain))
        ):
        for index_nstep in range(nstep):
            if step < (ntrain+nval) * (index_nstep + 1) and step >= (ntrain+nval) * (index_nstep):
                # Energy is shifted by the reference energy
                # to avoid the unsual weighting with forces in NequIP

                if harmonic_F:
                    from libs.lib_util import get_displacements, get_fc_ha, get_E_ha
                    displacements = get_displacements(traj[i]['atoms']['positions'], 'geometry.in.supercell')
                    F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
                    F_step = np.array(traj[i]['calculator']['forces']) - F_ha
                    E_ha = get_E_ha(displacements, F_ha)
                    E_step = np.array(traj[i]['calculator']['energy']) - E_gs - E_ha
                    
                else:
                    F_step = np.array(traj[i]['calculator']['forces'])
                    E_step = np.array(traj[i]['calculator']['energy']) - E_gs

                E_train[index_nstep].append(E_step);
                F_train[index_nstep].append(F_step);
                R_train[index_nstep].append(traj[i]['atoms']['positions']);
                z_train[index_nstep].append([atomic_numbers[item[1]] for item in traj[i]['atoms']['symbols'] for idx in range(item[0])]);
                CELL_train[index_nstep].append(traj[i]['atoms']['cell']);
                PBC_train[index_nstep].append(traj[i]['atoms']['pbc'])
                traj_idx.append(i)
                break

    # Split the sampled data into individual files for each subsampling set
    for index_nstep in range(nstep):
        E_train_store    = E_train[index_nstep]
        F_train_store    = F_train[index_nstep]
        R_train_store    = R_train[index_nstep]
        z_train_store    = z_train[index_nstep]
        CELL_train_store = CELL_train[index_nstep]
        PBC_train_store  = PBC_train[index_nstep]

        # Save each subsampling data
        npz_name = f'{workpath}/data-train_{index_nstep}.npz'
        np.savez(
            npz_name[:-4],
            E=np.array(E_train_store),
            F=np.array(F_train_store),
            R=np.array(R_train_store),
            z=np.array(z_train_store),
            CELL=np.array(CELL_train_store),
            PBC=np.array(PBC_train_store)
        )
    single_print('[npz]\tFinish the sampling process: data-train_*.npz')
    
    return traj_idx
    
    

def generate_npz_DFT_rand(
    traj, ntrain, nval, nstep, E_gs, index,
    temperature, pressure, workpath, traj_idx, harmonic_F
):
    """Function [generate_npz_DFT_rand]
    Generate training data sets
    by adding new data from the trajectory to previous training data sets.
    The main distinction from the function [generate_npz_DFT] is
    that this function returnsa list of randomly selected sample indices
    (traj_idx).

    Parameters:

    traj: ASE trajectory
        Training data will be randomly sampled from this trajectory
    ntrain: int
        The total number of training data
    nval: int
        The total number of validating data
    nstep: int
        The number of subsampling sets
    E_gs: float
        Reference total energy in units of eV/Unitcell
        to shift the total energies of the trajectory
        and avoid unusually high total energy values
        that may lead to unusual weightings with force values.
        Recommend to use the ground state total energy.
    index: int
        The index of AL interactive step

    temperature: float
        The desired temperature in units of Kelvin (K)
    output_format: str
        The format of output file that be read to add new data
        'aims.out' or 'trajectory.son'
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    workpath: str
        The path to the working directory
    traj_idx: list of int
        The list of randomly selected sample indices until previous step

    Returns:
    traj_idx: list of int
        The list of randomly selected sample indices until current step
    """

    # Prepare the empty list of propreties for each subsampling set
    E_train      = [[] for i in range(nstep)];
    F_train      = [[] for i in range(nstep)];
    R_train      = [[] for i in range(nstep)];
    z_train      = [[] for i in range(nstep)];
    CELL_train   = [[] for i in range(nstep)];
    PBC_train    = [[] for i in range(nstep)];
    
    # The total number of data we need for the training process
    total_ntrain = (ntrain+nval) * nstep

    # Check the existence of NPZ files containing previous data
    npz_check = [os.path.exists(f'{workpath}/data-train_{index_nstep}.npz')\
                 for index_nstep in range(nstep)];

    # For first run, of course there is no NPZ file
    if all(npz_check) == False: # If there is any missing NPZ file,
        del npz_check
        single_print(f'[npz]\tSample {nstep} different training data\n')

        # Randomly sample the new data except previously sampled ones
        for i, step in zip(
            random.sample(list(set(range(0,len(traj)))-set(traj_idx)),total_ntrain),
            tqdm(range(total_ntrain))
            ):
            # Collect these new data for each subsampling
            for index_nstep in range(nstep):
                if step < (ntrain+nval) * (index_nstep + 1) and step >= (ntrain+nval) * (index_nstep):
                    if harmonic_F:
                        from libs.lib_util import get_displacements, get_fc_ha, get_E_ha
                        displacements = get_displacements(traj[i]['atoms']['positions'], 'geometry.in.supercell')
                        F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
                        F_step = np.array(traj[i]['calculator']['forces']) - F_ha
                        E_ha = get_E_ha(displacements, F_ha)
                        E_step = np.array(traj[i]['calculator']['energy']) - E_gs - E_ha
                        
                    else:
                        F_step = np.array(traj[i]['calculator']['forces'])
                        E_step = np.array(traj[i]['calculator']['energy']) - E_gs

                    E_train[index_nstep].append(E_step);
                    F_train[index_nstep].append(F_step);
                    R_train[index_nstep].append(traj[i]['atoms']['positions']);
                    z_train[index_nstep].append([atomic_numbers[item[1]] for item in traj[i]['atoms']['symbols'] for idx in range(item[0])]);
                    CELL_train[index_nstep].append(traj[i]['atoms']['cell']);
                    PBC_train[index_nstep].append(traj[i]['atoms']['pbc'])
                    traj_idx.append(i)
                    break

        # Merge new data with previous data
        # Split the sampled data into individual files for each subsampling set   
        for index_nstep in range(nstep):
            npz_previous = f'./MODEL/{temperature}K-{pressure}bar_{index-1}'\
                           + f'/data-train_{index_nstep}.npz'
            
            # When there is previous data, merge them together
            if os.path.exists(npz_previous):
                data_train       = np.load(npz_previous)
                E_train_store    = np.concatenate\
                ((data_train['E'], E_train[index_nstep]), axis=0)
                F_train_store    = np.concatenate\
                ((data_train['F'], F_train[index_nstep]), axis=0)
                R_train_store    = np.concatenate\
                ((data_train['R'], R_train[index_nstep]), axis=0)
                z_train_store    = np.concatenate\
                ((data_train['z'], z_train[index_nstep]), axis=0)
                CELL_train_store = np.concatenate\
                ((data_train['CELL'], CELL_train[index_nstep]), axis=0)
                PBC_train_store  = np.concatenate\
                ((data_train['PBC'], PBC_train[index_nstep]), axis=0)

            # Path to new data
            npz_name = f'{workpath}/data-train_{index_nstep}.npz'
            # Save each subsampling data
            np.savez(
                npz_name[:-4],\
                E=np.array(E_train_store),\
                F=np.array(F_train_store),\
                R=np.array(R_train_store),\
                z=np.array(z_train_store),\
                CELL=np.array(CELL_train_store),\
                PBC=np.array(PBC_train_store)
            )
        single_print('[npz]\tFinish the sampling process: data-train_*.npz')
    else:
        single_print('[npz]\tFound all sampled training data: data-train_*.npz')

        
    return traj_idx