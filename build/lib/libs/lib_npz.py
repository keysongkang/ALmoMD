import os
import random
import numpy as np
from tqdm import tqdm
from ase.data   import atomic_numbers
from ase.io     import read as atoms_read

from libs.lib_util   import read_aims, single_print
import son


def generate_npz_temp(
    traj, ntrain, nval, nstep, E_gs, index, temperature,
    init_temp, step_temp, pressure, workpath, calculator
):
    E_train      = [[] for i in range(nstep)];
    F_train      = [[] for i in range(nstep)];
    R_train      = [[] for i in range(nstep)];
    z_train      = [[] for i in range(nstep)];
    CELL_train   = [[] for i in range(nstep)];
    PBC_train    = [[] for i in range(nstep)];
    stress_train = [[] for i in range(nstep)];
    
    total_ntrain = (ntrain+nval) * nstep
    npz_check = [os.path.exists(f'{workpath}/data-train_{index_nstep}.npz')\
                 for index_nstep in range(nstep)];

    if all(npz_check) == False:
        del npz_check
        single_print(f'Sample {nstep} different training data\n')
        for i, step in zip(random.sample(range(0,len(traj)),total_ntrain),\
                           tqdm(range(total_ntrain))):
            for index_nstep in range(nstep):
                if step < (ntrain+nval) * (index_nstep + 1):
                    atoms = traj[i]
                    atoms.calc = calculator
                    E_train[index_nstep].append(atoms.get_potential_energy() - E_gs);
                    F_train[index_nstep].append(atoms.get_forces());
                    R_train[index_nstep].append(atoms.get_positions());
                    z_train[index_nstep].append(atoms.numbers);
                    CELL_train[index_nstep].append(atoms.get_cell());
                    PBC_train[index_nstep].append(atoms.get_pbc());
                    break
                    
        for index_nstep in range(nstep):
            npz_previous = f'./data/{temperature}K-{pressure}bar_{index-1}'\
                           + f'/data-train_{index_nstep}.npz'
                
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
                stress_train_store = np.concatenate\
                ((data_train['stress'], stress_train[index_nstep]), axis=0)
            else:
                E_train_store    = E_train[index_nstep]
                F_train_store    = F_train[index_nstep]
                R_train_store    = R_train[index_nstep]
                z_train_store    = z_train[index_nstep]
                CELL_train_store = CELL_train[index_nstep]
                PBC_train_store  = PBC_train[index_nstep]
                stress_train_store = stress_train[index_nstep]
                
            npz_name = f'{workpath}/data-train_{index_nstep}.npz'
            np.savez(
                npz_name[:-4],\
                E=np.array(E_train_store),\
                F=np.array(F_train_store),\
                R=np.array(R_train_store),\
                z=np.array(z_train_store),\
                CELL=np.array(CELL_train_store),\
                PBC=np.array(PBC_train_store),\
                stress=np.array(stress_train_store)\
            )
        single_print('Finish the sampling process: data-train_*.npz')
    else:
        single_print('Found all sampled training data: data-train_*.npz')

        
def generate_npz_init_cont(
    traj, ntrain, nval, nstep, E_gs, index, temperature, final_temp,
    init_temp, step_temp, pressure, workpath, calculator
):
    E_train      = [[] for i in range(nstep)];
    F_train      = [[] for i in range(nstep)];
    R_train      = [[] for i in range(nstep)];
    z_train      = [[] for i in range(nstep)];
    CELL_train   = [[] for i in range(nstep)];
    PBC_train    = [[] for i in range(nstep)];
    stress_train = [[] for i in range(nstep)];
    
    total_ntrain = (ntrain+nval) * nstep
    npz_check = [os.path.exists(f'{workpath}/data-train_{index_nstep}.npz')\
                 for index_nstep in range(nstep)];

    if all(npz_check) == False:
        del npz_check
        single_print(f'Sample {nstep} different training data\n')
        for i, step in zip(random.sample(range(0,len(traj)),total_ntrain),\
                           tqdm(range(total_ntrain))):
            for index_nstep in range(nstep):
                if step < (ntrain+nval) * (index_nstep + 1):
                    atoms = traj[i]
                    atoms.calc = calculator
                    E_train[index_nstep].append(atoms.get_potential_energy() - E_gs);
                    F_train[index_nstep].append(atoms.get_forces());
                    R_train[index_nstep].append(atoms.get_positions());
                    z_train[index_nstep].append(atoms.numbers);
                    CELL_train[index_nstep].append(atoms.get_cell());
                    PBC_train[index_nstep].append(atoms.get_pbc());
                    break
                    
        for index_nstep in range(nstep):
            npz_previous = f'./data/{final_temp}K-{pressure}bar_converged'\
                           + f'/data-train_{index_nstep}.npz'
                
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
                stress_train_store = np.concatenate\
                ((data_train['stress'], stress_train[index_nstep]), axis=0)
            else:
                E_train_store    = E_train[index_nstep]
                F_train_store    = F_train[index_nstep]
                R_train_store    = R_train[index_nstep]
                z_train_store    = z_train[index_nstep]
                CELL_train_store = CELL_train[index_nstep]
                PBC_train_store  = PBC_train[index_nstep]
                stress_train_store = stress_train[index_nstep]
                
            npz_name = f'{workpath}/data-train_{index_nstep}.npz'
            np.savez(
                npz_name[:-4],\
                E=np.array(E_train_store),\
                F=np.array(F_train_store),\
                R=np.array(R_train_store),\
                z=np.array(z_train_store),\
                CELL=np.array(CELL_train_store),\
                PBC=np.array(PBC_train_store),\
                stress=np.array(stress_train_store)\
            )
        single_print('Finish the sampling process: data-train_*.npz')
    else:
        single_print('Found all sampled training data: data-train_*.npz')

        
def generate_npz_DFT_init(
    traj, ntrain, nval, nstep, E_gs, index, temperature,
    pressure, workpath
):
    E_train      = [[] for i in range(nstep)];
    F_train      = [[] for i in range(nstep)];
    R_train      = [[] for i in range(nstep)];
    z_train      = [[] for i in range(nstep)];
    CELL_train   = [[] for i in range(nstep)];
    PBC_train    = [[] for i in range(nstep)];
    
    total_ntrain = (ntrain+nval) * nstep

    single_print(f'Sample {nstep} different training data\n')
    for i, step in zip(random.sample(range(0,len(traj)),total_ntrain),\
                       tqdm(range(total_ntrain))):
        for index_nstep in range(nstep):
            if step < (ntrain+nval) * (index_nstep + 1):
                E_train[index_nstep].append(traj[i]['calculator']['energy'] - E_gs);
                F_train[index_nstep].append(traj[i]['calculator']['forces']);
                R_train[index_nstep].append(traj[i]['atoms']['positions']);
                z_train[index_nstep].append([atomic_numbers[item[1]] for item in traj[i]['atoms']['symbols'] for index in range(item[0])]);
                CELL_train[index_nstep].append(traj[i]['atoms']['cell']);
                PBC_train[index_nstep].append(traj[i]['atoms']['pbc'])
                break

    for index_nstep in range(nstep):
        E_train_store    = E_train[index_nstep]
        F_train_store    = F_train[index_nstep]
        R_train_store    = R_train[index_nstep]
        z_train_store    = z_train[index_nstep]
        CELL_train_store = CELL_train[index_nstep]
        PBC_train_store  = PBC_train[index_nstep]

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
    single_print('Finish the sampling process: data-train_*.npz')
    
    
def generate_npz_DFT(
    ntrain, nval, nstep, E_gs, index, temperature,
    output_format, pressure, workpath
):
    E_train      = [[] for i in range(nstep)];
    F_train      = [[] for i in range(nstep)];
    R_train      = [[] for i in range(nstep)];
    z_train      = [[] for i in range(nstep)];
    CELL_train   = [[] for i in range(nstep)];
    PBC_train    = [[] for i in range(nstep)];
    
    total_ntrain = (ntrain+nval) * nstep
    npz_check = [os.path.exists(f'{workpath}/data-train_{index_nstep}.npz')\
                 for index_nstep in range(nstep)];

    if all(npz_check) == False:
        del npz_check
        single_print(f'Sample {nstep} different training data\n')
        for i, step in zip(random.sample(range(0,total_ntrain),total_ntrain),\
                           tqdm(range(total_ntrain))):
            for index_nstep in range(nstep):
                if step < (ntrain+nval) * (index_nstep + 1):
                    if output_format == 'aims.out':
                        atoms, atoms_potE, atoms_forces = read_aims(f'./calc/{temperature}K-{pressure}bar_{index}/{i}/aims/calculations/aims.out')
                        E_train[index_nstep].append(atoms_potE - E_gs);
                        F_train[index_nstep].append(atoms_forces);
                        R_train[index_nstep].append(atoms.get_positions());
                        z_train[index_nstep].append(atoms.numbers);
                        CELL_train[index_nstep].append(atoms.get_cell());
                        PBC_train[index_nstep].append(atoms.get_pbc());
                        break
                    elif output_format == 'trajectory.son':
                        metadata, data = son.load(f'./calc/{temperature}K-{pressure}bar_{index}/{i}/aims/trajectory.son')
                        atom_numbers = []
                        for items in data[0]['atoms']['symbols']:
                            for jndex in range(items[0]):
                                atom_numbers.append(atomic_numbers[items[1]])
                        E_train[index_nstep].append(data[0]['calculator']['energy'] - E_gs);
                        F_train[index_nstep].append(data[0]['calculator']['forces']);
                        R_train[index_nstep].append(data[0]['atoms']['positions']);
                        z_train[index_nstep].append(np.array(atom_numbers));
                        CELL_train[index_nstep].append(data[0]['atoms']['cell']);
                        PBC_train[index_nstep].append(data[0]['atoms']['pbc']);
                        break
                    else:
                        'You need to define the output format.'
                    
        for index_nstep in range(nstep):
            npz_previous = f'./data/{temperature}K-{pressure}bar_{index-1}'\
                           + f'/data-train_{index_nstep}.npz'
                
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
            else:
                E_train_store    = E_train[index_nstep]
                F_train_store    = F_train[index_nstep]
                R_train_store    = R_train[index_nstep]
                z_train_store    = z_train[index_nstep]
                CELL_train_store = CELL_train[index_nstep]
                PBC_train_store  = PBC_train[index_nstep]
                
            npz_name = f'{workpath}/data-train_{index_nstep}.npz'
            np.savez(
                npz_name[:-4],\
                E=np.array(E_train_store),\
                F=np.array(F_train_store),\
                R=np.array(R_train_store),\
                z=np.array(z_train_store),\
                CELL=np.array(CELL_train_store),\
                PBC=np.array(PBC_train_store)
            )
        single_print('Finish the sampling process: data-train_*.npz')
    else:
        single_print('Found all sampled training data: data-train_*.npz')

        
        
def generate_npz_DFT_rand_init(
    traj, ntrain, nval, nstep, E_gs, index, temperature,
    pressure, workpath
):
    E_train      = [[] for i in range(nstep)];
    F_train      = [[] for i in range(nstep)];
    R_train      = [[] for i in range(nstep)];
    z_train      = [[] for i in range(nstep)];
    CELL_train   = [[] for i in range(nstep)];
    PBC_train    = [[] for i in range(nstep)];
    traj_idx     = [];
    
    total_ntrain = (ntrain+nval) * nstep

    single_print(f'Sample {nstep} different training data\n')
    for i, step in zip(random.sample(range(0,len(traj)),total_ntrain),\
                       tqdm(range(total_ntrain))):
        for index_nstep in range(nstep):
            if step < (ntrain+nval) * (index_nstep + 1):
                E_train[index_nstep].append(traj[i]['calculator']['energy'] - E_gs);
                F_train[index_nstep].append(traj[i]['calculator']['forces']);
                R_train[index_nstep].append(traj[i]['atoms']['positions']);
                z_train[index_nstep].append([atomic_numbers[item[1]] for item in traj[i]['atoms']['symbols'] for index in range(item[0])]);
                CELL_train[index_nstep].append(traj[i]['atoms']['cell']);
                PBC_train[index_nstep].append(traj[i]['atoms']['pbc'])
                traj_idx.append(i)
                break

    for index_nstep in range(nstep):
        E_train_store    = E_train[index_nstep]
        F_train_store    = F_train[index_nstep]
        R_train_store    = R_train[index_nstep]
        z_train_store    = z_train[index_nstep]
        CELL_train_store = CELL_train[index_nstep]
        PBC_train_store  = PBC_train[index_nstep]

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
    single_print('Finish the sampling process: data-train_*.npz')
    
    return traj_idx
    
    
def generate_npz_DFT_rand(
    traj, ntrain, nval, nstep, E_gs, index, temperature,
    pressure, workpath, traj_idx
):
    E_train      = [[] for i in range(nstep)];
    F_train      = [[] for i in range(nstep)];
    R_train      = [[] for i in range(nstep)];
    z_train      = [[] for i in range(nstep)];
    CELL_train   = [[] for i in range(nstep)];
    PBC_train    = [[] for i in range(nstep)];
    
    total_ntrain = (ntrain+nval) * nstep
    npz_check = [os.path.exists(f'{workpath}/data-train_{index_nstep}.npz')\
                 for index_nstep in range(nstep)];

    if all(npz_check) == False:
        del npz_check
        single_print(f'Sample {nstep} different training data\n')
        for i, step in zip(random.sample(list(set(range(0,len(traj)))-set(traj_idx)),total_ntrain),\
                           tqdm(range(total_ntrain))):
            for index_nstep in range(nstep):
                if step < (ntrain+nval) * (index_nstep + 1):
                    E_train[index_nstep].append(traj[i]['calculator']['energy'] - E_gs);
                    F_train[index_nstep].append(traj[i]['calculator']['forces']);
                    R_train[index_nstep].append(traj[i]['atoms']['positions']);
                    z_train[index_nstep].append([atomic_numbers[item[1]] for item in traj[i]['atoms']['symbols'] for index in range(item[0])]);
                    CELL_train[index_nstep].append(traj[i]['atoms']['cell']);
                    PBC_train[index_nstep].append(traj[i]['atoms']['pbc'])
                    traj_idx.append(i)
                    break
                    
        for index_nstep in range(nstep):
            npz_previous = f'./data/{temperature}K-{pressure}bar_{index-1}'\
                           + f'/data-train_{index_nstep}.npz'
                
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
            else:
                E_train_store    = E_train[index_nstep]
                F_train_store    = F_train[index_nstep]
                R_train_store    = R_train[index_nstep]
                z_train_store    = z_train[index_nstep]
                CELL_train_store = CELL_train[index_nstep]
                PBC_train_store  = PBC_train[index_nstep]
                
            npz_name = f'{workpath}/data-train_{index_nstep}.npz'
            np.savez(
                npz_name[:-4],\
                E=np.array(E_train_store),\
                F=np.array(F_train_store),\
                R=np.array(R_train_store),\
                z=np.array(z_train_store),\
                CELL=np.array(CELL_train_store),\
                PBC=np.array(PBC_train_store)
            )
        single_print('Finish the sampling process: data-train_*.npz')
    else:
        single_print('Found all sampled training data: data-train_*.npz')

        
    return traj_idx