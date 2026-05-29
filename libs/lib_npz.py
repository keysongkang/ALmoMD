import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
from ase.data import atomic_numbers
from ase.io.trajectory import Trajectory
from tqdm import tqdm
from vibes import son

from libs.lib_util import read_aims, single_print, Structure


@dataclass
class Sample:
    energy: float
    forces: np.ndarray
    positions: np.ndarray
    atomic_numbers: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    sigma_forces: np.ndarray
    stress: Optional[np.ndarray] = None


def _needs_stress(inputs):
    return inputs.ensemble == 'NPTisoiso' or inputs.train_stress


def _new_sets(inputs):
    keys = ('E', 'F', 'R', 'z', 'CELL', 'PBC')
    sets = {key: [[] for _ in range(inputs.nstep)] for key in keys}
    if inputs.npz_sigma:
        sets['sigma'] = [[] for _ in range(inputs.nstep)]
    if _needs_stress(inputs):
        sets['stress'] = [[] for _ in range(inputs.nstep)]
    return sets


def _cell_keys(inputs):
    if inputs.MLIP == 'so3krates':
        return 'unit_cell', 'pbc'
    return 'CELL', 'PBC'


def _son_numbers(symbols):
    return np.array([
        atomic_numbers[symbol]
        for count, symbol in symbols
        for _ in range(count)
    ])


def _apply_harmonic(inputs, positions, energy, forces):
    forces = np.array(forces)
    energy = np.array(energy)
    if not inputs.harmonic_F:
        return energy - inputs.E_gs, forces

    from libs.lib_util import get_displacements, get_fc_ha, get_E_ha
    displacements = get_displacements(positions, 'geometry.in.supercell')
    F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
    E_ha = get_E_ha(displacements, F_ha)
    return energy - inputs.E_gs - E_ha, forces - F_ha


def _sample_from_son(inputs, frame):
    atoms = frame['atoms']
    calc = frame['calculator']
    energy, forces = _apply_harmonic(
        inputs,
        atoms['positions'],
        calc['energy'],
        calc['forces'],
    )
    return Sample(
        energy=energy,
        forces=forces,
        positions=np.array(atoms['positions']),
        atomic_numbers=_son_numbers(atoms['symbols']),
        cell=np.array(atoms['cell']),
        pbc=np.array(atoms['pbc']),
        sigma_forces=np.array(calc['forces']),
        stress=np.array(calc['stress']) if _needs_stress(inputs) and 'stress' in calc else None,
    )


def _sample_from_aims(inputs, path):
    atoms, atoms_potE, atoms_forces = read_aims(path)
    atoms = Structure.from_atoms(atoms)
    energy, forces = _apply_harmonic(
        inputs,
        atoms.get_positions(),
        atoms_potE,
        atoms_forces,
    )
    stress = None
    if _needs_stress(inputs):
        try:
            stress = atoms.get_stress(voigt=False)
        except Exception:
            stress = None
    return Sample(
        energy=energy,
        forces=forces,
        positions=np.array(atoms.get_positions()),
        atomic_numbers=np.array(atoms.numbers),
        cell=np.array(atoms.get_cell()),
        pbc=np.array(atoms.get_pbc()),
        sigma_forces=forces,
        stress=stress,
    )


def _sample_from_nequip(inputs, path):
    data = Trajectory(path)
    atoms = data[0]
    energy, forces = _apply_harmonic(
        inputs,
        atoms.get_positions(),
        atoms.get_potential_energy(),
        atoms.get_forces(),
    )
    stress = None
    if _needs_stress(inputs):
        try:
            stress = atoms.get_stress(voigt=False)
        except Exception:
            stress = None
    return Sample(
        energy=energy,
        forces=forces,
        positions=np.array(atoms.get_positions()),
        atomic_numbers=np.array(atoms.get_atomic_numbers()),
        cell=np.array(atoms.get_cell()),
        pbc=np.array(atoms.get_pbc()),
        sigma_forces=forces,
        stress=stress,
    )


def _append_sample(inputs, sets, index_nstep, sample):
    sets['E'][index_nstep].append(sample.energy)
    sets['F'][index_nstep].append(sample.forces)
    sets['R'][index_nstep].append(sample.positions)
    sets['z'][index_nstep].append(sample.atomic_numbers)
    sets['CELL'][index_nstep].append(sample.cell)
    sets['PBC'][index_nstep].append(sample.pbc)

    if inputs.npz_sigma:
        from libs.lib_util import eval_sigma
        sets['sigma'][index_nstep].append(
            eval_sigma(
                struc_step_forces=sample.sigma_forces,
                struc_step_positions=sample.positions,
                al_type='sigma',
            )
        )

    if _needs_stress(inputs):
        sets['stress'][index_nstep].append(sample.stress)


def _save_sets(inputs, sets, workpath, previous_dir=None, include_zero_sigma=False):
    cell_key, pbc_key = _cell_keys(inputs)
    for index_nstep in range(inputs.nstep):
        arrays = {
            'E': np.array(sets['E'][index_nstep]),
            'F': np.array(sets['F'][index_nstep]),
            'R': np.array(sets['R'][index_nstep]),
            'z': np.array(sets['z'][index_nstep]),
            cell_key: np.array(sets['CELL'][index_nstep]),
            pbc_key: np.array(sets['PBC'][index_nstep]),
        }

        if inputs.npz_sigma:
            arrays['sigma'] = np.array(sets['sigma'][index_nstep])
        elif include_zero_sigma:
            arrays['sigma'] = np.zeros_like(arrays['E'])

        if _needs_stress(inputs):
            arrays['stress'] = np.array(sets['stress'][index_nstep])

        if previous_dir is not None:
            previous_npz = f'{previous_dir}/data-train_{index_nstep}.npz'
            if os.path.exists(previous_npz):
                with np.load(previous_npz) as previous:
                    merged = {
                        'E': np.concatenate((previous['E'], arrays['E']), axis=0),
                        'F': np.concatenate((previous['F'], arrays['F']), axis=0),
                        'R': np.concatenate((previous['R'], arrays['R']), axis=0),
                        'z': np.concatenate((previous['z'], arrays['z']), axis=0),
                        cell_key: np.concatenate((previous[cell_key], arrays[cell_key]), axis=0),
                        pbc_key: np.concatenate((previous[pbc_key], arrays[pbc_key]), axis=0),
                    }
                    if inputs.npz_sigma and 'sigma' in previous:
                        merged['sigma'] = np.concatenate((previous['sigma'], arrays['sigma']), axis=0)
                    elif include_zero_sigma:
                        merged['sigma'] = arrays['sigma']
                    if _needs_stress(inputs) and 'stress' in previous:
                        merged['stress'] = np.concatenate((previous['stress'], arrays['stress']), axis=0)
                    arrays = merged

        np.savez(f'{workpath}/data-train_{index_nstep}.npz', **arrays)


def _subsample_index(step, per_set, nstep):
    index_nstep = step // per_set
    return index_nstep if 0 <= index_nstep < nstep else None


def _collect_from_son_trajectory(inputs, traj, indices, per_set, sets, traj_idx=None):
    for i, step in zip(indices, tqdm(range(len(indices)))):
        index_nstep = _subsample_index(step, per_set, inputs.nstep)
        if index_nstep is None:
            continue
        _append_sample(inputs, sets, index_nstep, _sample_from_son(inputs, traj[i]))
        if traj_idx is not None:
            traj_idx.append(i)


def _all_npz_exist(inputs, workpath):
    return all(
        os.path.exists(f'{workpath}/data-train_{index_nstep}.npz')
        for index_nstep in range(inputs.nstep)
    )


def generate_npz_DFT_init(inputs, traj, workpath):
    """Generate the initial training data sets from trajectory."""

    sets = _new_sets(inputs)
    per_set = inputs.ntrain_init + inputs.nval_init

    single_print(f'[npz]\tSample {inputs.nstep} different training data\n')
    indices = random.sample(range(len(traj)), inputs.ntotal_init)
    _collect_from_son_trajectory(inputs, traj, indices, per_set, sets)
    _save_sets(inputs, sets, workpath, include_zero_sigma=True)

    single_print('[npz]\tFinish the sampling process: data-train_*.npz')


def generate_npz_DFT(inputs, workpath):
    """Generate training data sets by adding new DFT results."""

    if _all_npz_exist(inputs, workpath):
        single_print('[npz]\tFound all sampled training data: data-train_*.npz')
        return

    sets = _new_sets(inputs)
    per_set = inputs.ntrain + inputs.nval

    single_print(f'[npz]\tSample {inputs.nstep} different training data\n')
    indices = random.sample(range(inputs.ntotal), inputs.ntotal)
    condition = f'{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'

    for i, step in zip(indices, tqdm(range(inputs.ntotal))):
        index_nstep = _subsample_index(step, per_set, inputs.nstep)
        if index_nstep is None:
            continue

        if inputs.output_format == 'aims.out':
            sample = _sample_from_aims(
                inputs,
                f'./CALC/{condition}/{i}/aims/calculations/aims.out',
            )
        elif inputs.output_format == 'trajectory.son':
            _, data = son.load(f'./CALC/{condition}/{i}/aims/trajectory.son')
            sample = _sample_from_son(inputs, data[0])
        elif inputs.output_format == 'nequip':
            sample = _sample_from_nequip(inputs, f'./CALC/{condition}/{i}/geometry.traj')
        else:
            single_print('[npz]\tYou need to define the output format.')
            continue

        _append_sample(inputs, sets, index_nstep, sample)

    previous_dir = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index-1}'
    _save_sets(inputs, sets, workpath, previous_dir=previous_dir)

    single_print('[npz]\tFinish the sampling process: data-train_*.npz')


def generate_npz_DFT_rand_init(inputs, traj, ntrain, nval, workpath):
    """Generate initial random-sampling training data and return sampled indices."""

    sets = _new_sets(inputs)
    traj_idx = []
    per_set = inputs.ntrain + inputs.nval

    single_print(f'[npz]\tSample {inputs.nstep} different training data\n')
    indices = random.sample(range(len(traj)), inputs.ntotal)
    _collect_from_son_trajectory(inputs, traj, indices, per_set, sets, traj_idx)
    _save_sets(inputs, sets, workpath)

    single_print('[npz]\tFinish the sampling process: data-train_*.npz')
    return traj_idx


def generate_npz_DFT_rand(inputs, traj, workpath, traj_idx):
    """Generate random-sampling training data and return all sampled indices."""

    if _all_npz_exist(inputs, workpath):
        single_print('[npz]\tFound all sampled training data: data-train_*.npz')
        return traj_idx

    sets = _new_sets(inputs)
    per_set = inputs.ntrain + inputs.nval

    single_print(f'[npz]\tSample {inputs.nstep} different training data\n')
    available = list(set(range(len(traj))) - set(traj_idx))
    indices = random.sample(available, inputs.ntotal)
    _collect_from_son_trajectory(inputs, traj, indices, per_set, sets, traj_idx)

    previous_dir = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index-1}'
    _save_sets(inputs, sets, workpath, previous_dir=previous_dir)

    single_print('[npz]\tFinish the sampling process: data-train_*.npz')
    return traj_idx
