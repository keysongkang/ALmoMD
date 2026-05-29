import os
import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

import ase.units as units
import numpy as np
import pandas as pd
from ase.build import make_supercell
from ase.io import read as atoms_read
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

import torch

from libs.lib_criteria import eval_uncert, get_criteria, get_criteria_prob, uncert_strconvter
from libs.lib_MD_util import get_forces, get_MDinfo_temp, get_masses
from libs.lib_util import Structure, single_print

torch.set_default_dtype(torch.float64)


MODE_NORMAL = 'normal'
MODE_META = 'meta'
MODE_BIAS = 'bias'
MODE_TEMP = 'temp'
MODE_BIAS_TEMP = 'bias_temp'


@dataclass
class ForceState:
    forces: np.ndarray
    step_std: Optional[np.ndarray] = None


def cont_NVTLangevin(
    inputs, struc: Structure, timestep, temperature, calculator, E_ref,
    MD_index, MD_step_index, signal_uncert=False, signal_append=True,
    fix_com=True,
):
    _run_cont_nvtlangevin(
        MODE_NORMAL, inputs, struc, timestep, temperature, calculator, E_ref,
        MD_index, MD_step_index, signal_uncert, signal_append, fix_com,
    )


def cont_NVTLangevin_meta(
    inputs, struc, timestep, temperature, calculator, E_ref,
    MD_index, MD_step_index, signal_uncert=False, signal_append=True,
    fix_com=True,
):
    _run_cont_nvtlangevin(
        MODE_META, inputs, struc, timestep, temperature, calculator, E_ref,
        MD_index, MD_step_index, signal_uncert, signal_append, fix_com,
    )


def cont_NVTLangevin_bias(
    inputs, struc: Structure, timestep, temperature, calculator, E_ref,
    MD_index, MD_step_index, signal_uncert=False, signal_append=True,
    fix_com=True,
):
    _run_cont_nvtlangevin(
        MODE_BIAS, inputs, struc, timestep, temperature, calculator, E_ref,
        MD_index, MD_step_index, signal_uncert, signal_append, fix_com,
    )


def cont_NVTLangevin_temp(
    inputs, struc, timestep, temperature, calculator, E_ref,
    MD_index, MD_step_index, signal_uncert=False, signal_append=True,
    fix_com=True,
):
    _run_cont_nvtlangevin(
        MODE_TEMP, inputs, struc, timestep, temperature, calculator, E_ref,
        MD_index, MD_step_index, signal_uncert, signal_append, fix_com,
    )


def cont_NVTLangevin_bias_temp(
    inputs, struc, timestep, temperature, calculator, E_ref,
    MD_index, MD_step_index, signal_uncert=False, signal_append=True,
    fix_com=True,
):
    _run_cont_nvtlangevin(
        MODE_BIAS_TEMP, inputs, struc, timestep, temperature, calculator, E_ref,
        MD_index, MD_step_index, signal_uncert, signal_append, fix_com,
    )


def _run_cont_nvtlangevin(
    mode, inputs, struc, timestep, temperature, calculator, E_ref,
    MD_index, MD_step_index, signal_uncert, signal_append, fix_com,
):
    condition = f'{inputs.temperature}K-{inputs.pressure}bar'
    trajectory = f'TEMPORARY/temp-{condition}_{inputs.index}.traj'
    logfile = f'TEMPORARY/temp-{condition}_{inputs.index}.log'
    criteria_collected = get_criteria(
        inputs.temperature, inputs.pressure, inputs.index,
        inputs.steps_init, inputs.al_type,
    )

    if os.path.exists(trajectory):
        traj_temp = Trajectory(trajectory)
        struc = traj_temp[-1]
        if mode in (MODE_NORMAL, MODE_BIAS):
            struc = Structure.from_atoms(struc)
        MD_step_index = len(traj_temp)
        del traj_temp

    file_traj = _open_temp_trajectory(
        trajectory, logfile, struc, inputs, calculator, E_ref,
        MD_step_index, signal_uncert,
    )
    write_traj = TrajectoryWriter(
        filename=f'TRAJ/traj-{condition}_{inputs.index + 1}.traj',
        mode='a',
    )

    force_state = _calculate_force_state(mode, struc, inputs, calculator, E_ref, criteria_collected)

    while (MD_index < inputs.ntotal) or (
        inputs.calc_type == 'period'
        and MD_step_index < inputs.nperiod * inputs.loginterval
    ):
        struc = _maybe_meta_restart(mode, inputs, struc, condition, MD_index)

        natoms = len(struc)
        masses = get_masses(struc.get_masses(), natoms)
        velocity = struc.get_velocities()

        c1, c2, c3, c4, c5 = _langevin_coefficients(
            timestep, temperature, inputs.friction, masses,
        )
        if mode in (MODE_TEMP, MODE_BIAS_TEMP):
            _apply_temperature_boost(
                mode, inputs, timestep, temperature, masses, c3, c4, c5,
                force_state.step_std, criteria_collected, MD_step_index,
            )

        xi = np.random.standard_normal(size=(natoms, 3))
        eta = np.random.standard_normal(size=(natoms, 3))
        rnd_pos = c5 * eta
        rnd_vel = c3 * xi - c4 * eta

        if fix_com:
            rnd_pos -= rnd_pos.sum(axis=0) / natoms
            rnd_vel -= (rnd_vel * masses).sum(axis=0) / (masses * natoms)

        velocity += c1 * force_state.forces / masses - c2 * velocity + rnd_vel

        position = struc.get_positions()
        struc.set_positions(position + timestep * velocity + rnd_pos)
        _invalidate_cached_structure_data(struc)

        velocity = (struc.get_positions() - position - rnd_pos) / timestep
        force_state = _calculate_force_state(mode, struc, inputs, calculator, E_ref, criteria_collected)
        velocity += c1 * force_state.forces / masses - c2 * velocity + rnd_vel
        struc.set_momenta(velocity * masses)

        if (MD_step_index + 1) % inputs.loginterval == 0:
            MD_index = _record_langevin_step(
                mode, inputs, struc, timestep, calculator, E_ref,
                criteria_collected, condition, logfile, file_traj, write_traj,
                MD_index, MD_step_index, signal_uncert,
            )

        MD_step_index += 1

    file_traj.close()
    write_traj.close()


def _open_temp_trajectory(
    trajectory, logfile, struc, inputs, calculator, E_ref,
    MD_step_index, signal_uncert,
):
    if MD_step_index != 0:
        return TrajectoryWriter(filename=trajectory, mode='a')

    file_traj = TrajectoryWriter(filename=trajectory, mode='w')
    file_traj.write(atoms=struc)
    _initialize_log(logfile, struc, inputs, calculator, E_ref, signal_uncert)
    return file_traj


def _initialize_log(logfile, struc, inputs, calculator, E_ref, signal_uncert):
    with open(logfile, 'w') as file_log:
        file_log.write('Time[ps]   \tEtot[eV]   \tEpot[eV]    \tEkin[eV]   \tTemperature[K]')
        if signal_uncert:
            file_log.write(
                '\tUncertAbs_E\tUncertRel_E\tUncertAbs_F\tUncertRel_F\t'
                'UncertAbs_S\tUncertRel_S\tS_average\n'
            )
        else:
            file_log.write('\n')

    info_TE, info_PE, info_KE, info_T = get_MDinfo_temp(
        struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F, E_ref,
    )
    if signal_uncert:
        uncerts, _, S_step = eval_uncert(
            struc, inputs.nstep, inputs.nmodel, E_ref,
            calculator, inputs.al_type, inputs.harmonic_F,
        )

    with open(logfile, 'a') as file_log:
        file_log.write(_format_md_log_line(0.0, info_TE, info_PE, info_KE, info_T))
        if signal_uncert:
            file_log.write(
                '      \t'
                + uncert_strconvter(uncerts.UncertAbs_E) + '\t'
                + uncert_strconvter(uncerts.UncertRel_E) + '\t'
                + uncert_strconvter(uncerts.UncertAbs_F) + '\t'
                + uncert_strconvter(uncerts.UncertRel_F) + '\t'
                + uncert_strconvter(uncerts.UncertAbs_S) + '\t'
                + uncert_strconvter(uncerts.UncertRel_S) + '\t'
                + uncert_strconvter(S_step) + '\n'
            )
        else:
            file_log.write('\n')


def _maybe_meta_restart(mode, inputs, struc, condition, MD_index):
    if (
        mode not in (MODE_NORMAL, MODE_META)
        or not inputs.meta_restart
        or not os.path.exists('start.in')
        or inputs.ensemble != 'NVTLangevin_meta'
        or MD_index == 0
    ):
        return struc

    uncert_file = f'UNCERT/uncertainty-{condition}_{inputs.index}.txt'
    uncert_data = pd.read_csv(uncert_file, index_col=False, delimiter='\t')
    if np.array(uncert_data.loc[:, 'S_average'])[-1] <= inputs.meta_r_crtria:
        return struc

    single_print('[MLMD] Read a configuration from start.in')
    struc_init = atoms_read('start.in', format='aims')
    restarted = make_supercell(struc_init, inputs.supercell_init)
    if mode == MODE_NORMAL:
        restarted = Structure.from_atoms(restarted)
    MaxwellBoltzmannDistribution(
        restarted, temperature_K=inputs.temperature * 1.5, force_temp=True,
    )
    return restarted


def _langevin_coefficients(timestep, temperature, friction, masses):
    sigma = np.sqrt(2 * temperature * friction / masses)
    c1 = timestep / 2.0 - timestep * timestep * friction / 8.0
    c2 = timestep * friction / 2.0 - timestep * timestep * friction * friction / 8.0
    c3 = np.sqrt(timestep) * sigma / 2.0 - timestep**1.5 * friction * sigma / 8.0
    c5 = timestep**1.5 * sigma / (2 * np.sqrt(3))
    c4 = friction / 2.0 * c5
    return c1, c2, c3, c4, c5


def _apply_temperature_boost(
    mode, inputs, timestep, temperature, masses, c3, c4, c5,
    step_std, criteria, MD_step_index,
):
    if step_std is None:
        return

    idx_atom = inputs.idx_atom
    if inputs.al_type == 'force_max':
        center = criteria.Un_Abs_F_avg_i
        spread = criteria.Un_Abs_F_std_i
    elif inputs.al_type == 'energy_max' and mode == MODE_BIAS_TEMP:
        center = criteria.Un_Abs_Ea_avg_i
        spread = criteria.Un_Abs_Ea_std_i
    elif inputs.al_type == 'energy_max':
        center = criteria.Un_Abs_E_avg_i
        spread = criteria.Un_Abs_E_std_i
    else:
        temp_ratio = 0.0
        _set_boosted_atom_coefficients(inputs, timestep, temperature, masses, c3, c4, c5, temp_ratio)
        return

    delta = step_std[idx_atom] - center
    temp_ratio = 1.0 if delta < 0 else np.exp(-0.5 * (delta / spread) ** 2)
    single_print(f'Step {MD_step_index}; temp activate (atom {idx_atom}):{temp_ratio}')
    _set_boosted_atom_coefficients(inputs, timestep, temperature, masses, c3, c4, c5, temp_ratio)


def _set_boosted_atom_coefficients(inputs, timestep, temperature, masses, c3, c4, c5, temp_ratio):
    idx_atom = inputs.idx_atom
    sigma_elem = np.sqrt(
        2 * (temperature + temp_ratio * (inputs.temp_factor * units.kB))
        * inputs.friction / masses[idx_atom][0]
    )
    c3_elem = np.sqrt(timestep) * sigma_elem / 2.0 - timestep**1.5 * inputs.friction * sigma_elem / 8.0
    c5_elem = timestep**1.5 * sigma_elem / (2 * np.sqrt(3))
    c4_elem = inputs.friction / 2.0 * c5_elem

    c3[idx_atom] = c3_elem
    c4[idx_atom] = c4_elem
    c5[idx_atom] = c5_elem


def _record_langevin_step(
    mode, inputs, struc, timestep, calculator, E_ref, criteria_collected,
    condition, logfile, file_traj, write_traj, MD_index, MD_step_index,
    signal_uncert,
):
    uncerts, Epot_step, S_step = eval_uncert(
        struc, inputs.nstep, inputs.nmodel, E_ref,
        calculator, inputs.al_type, inputs.harmonic_F,
    )
    criteria = get_criteria_prob(inputs, Epot_step, uncerts, criteria_collected)

    if random.random() < criteria:
        accept = 'Accepted'
        MD_index += 1
        write_traj.write(atoms=struc)
    else:
        accept = 'Vetoed'

    info_TE, info_PE, info_KE, info_T = get_MDinfo_temp(
        struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F, E_ref,
    )
    _append_uncertainty(
        condition, inputs.index, info_T, uncerts, Epot_step,
        S_step, MD_index, criteria, accept,
    )
    _append_md_log(
        mode, logfile, timestep, MD_step_index, inputs.loginterval,
        info_TE, info_PE, info_KE, info_T, uncerts, S_step, signal_uncert,
    )
    file_traj.write(atoms=struc)
    return MD_index


def _append_uncertainty(
    condition, index, temperature, uncerts, Epot_step, S_step,
    MD_index, criteria, accept,
):
    with open(f'UNCERT/uncertainty-{condition}_{index}.txt', 'a') as trajfile:
        trajfile.write(
            '{:.5e}'.format(Decimal(str(temperature))) + '\t'
            + uncert_strconvter(uncerts.UncertAbs_E) + '\t'
            + uncert_strconvter(uncerts.UncertRel_E) + '\t'
            + uncert_strconvter(uncerts.UncertAbs_F) + '\t'
            + uncert_strconvter(uncerts.UncertRel_F) + '\t'
            + uncert_strconvter(uncerts.UncertAbs_S) + '\t'
            + uncert_strconvter(uncerts.UncertRel_S) + '\t'
            + uncert_strconvter(Epot_step) + '\t'
            + uncert_strconvter(S_step) + '\t'
            + str(MD_index) + '          \t'
            + '{:.5e}'.format(Decimal(str(criteria))) + '\t'
            + str(accept) + '   \n'
        )


def _append_md_log(
    mode, logfile, timestep, MD_step_index, loginterval,
    info_TE, info_PE, info_KE, info_T, uncerts, S_step, signal_uncert,
):
    sim_step = MD_step_index if mode == MODE_NORMAL else MD_step_index + loginterval
    simtime = timestep * sim_step / units.fs / 1000
    with open(logfile, 'a') as file_log:
        file_log.write(_format_md_log_line(simtime, info_TE, info_PE, info_KE, info_T))
        if signal_uncert:
            file_log.write(
                '      \t'
                + uncert_strconvter(uncerts.UncertAbs_E) + '\t'
                + uncert_strconvter(uncerts.UncertRel_E) + '\t'
                + uncert_strconvter(uncerts.UncertAbs_F) + '\t'
                + uncert_strconvter(uncerts.UncertRel_F) + '\t'
                + uncert_strconvter(uncerts.UncertAbs_S) + '\t'
                + uncert_strconvter(uncerts.UncertRel_S) + '\t'
                + uncert_strconvter(S_step) + '\n'
            )
        else:
            file_log.write('\n')


def _format_md_log_line(simtime, info_TE, info_PE, info_KE, info_T):
    return (
        '{:.5f}'.format(Decimal(str(simtime))) + '   \t'
        + '{:.5e}'.format(Decimal(str(info_TE))) + '\t'
        + '{:.5e}'.format(Decimal(str(info_PE))) + '\t'
        + '{:.5e}'.format(Decimal(str(info_KE))) + '\t'
        + '{:.2f}'.format(Decimal(str(info_T)))
    )


def _calculate_force_state(mode, struc, inputs, calculator, E_ref, criteria):
    if mode == MODE_NORMAL:
        return ForceState(
            get_forces(
                struc, inputs.nstep, inputs.nmodel, calculator,
                inputs.harmonic_F, inputs.anharmonic_F,
            )
        )
    if mode == MODE_META:
        return ForceState(
            get_forces_meta(
                struc, inputs.nstep, inputs.nmodel, calculator,
                inputs.harmonic_F, inputs.anharmonic_F,
                criteria, inputs.meta_Ediff, E_ref,
            )
        )
    if mode == MODE_BIAS:
        return ForceState(
            get_forces_bias(
                struc, inputs.nstep, inputs.nmodel, calculator,
                inputs.harmonic_F, inputs.anharmonic_F,
                criteria, inputs.bias_A, inputs.bias_B, inputs.idx_atom, E_ref,
            )
        )
    if mode == MODE_TEMP:
        forces, step_std = get_forces_temp(
            struc, inputs.nstep, inputs.nmodel, calculator,
            inputs.harmonic_F, inputs.anharmonic_F,
            criteria, inputs.al_type, E_ref,
        )
        return ForceState(forces, step_std)
    if mode == MODE_BIAS_TEMP:
        forces, step_std = get_forces_bias_temp(
            struc, inputs.nstep, inputs.nmodel, calculator,
            inputs.harmonic_F, inputs.anharmonic_F,
            criteria, inputs.al_type, inputs.bias_A, inputs.bias_B,
            inputs.idx_atom, E_ref,
        )
        return ForceState(forces, step_std)
    raise ValueError(f'Unknown NVT Langevin continuation mode: {mode}')


def get_forces_meta(
    struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F,
    criteria, meta_Ediff, E_ref,
):
    if not isinstance(calculator, list):
        struc.calc = calculator
        return struc.get_forces(md=True)

    energies, forces = _collect_total_energies_and_forces(
        struc, nstep, nmodel, calculator, E_ref,
    )
    force_avg = np.average(forces, axis=0)
    force_norm_std = _force_norm_std(forces, force_avg)
    uncert_coeff = np.zeros(len(struc.get_atomic_numbers()))
    uncert_max_idx = np.argmax(force_norm_std)

    from libs.lib_util import get_displacements, get_E_ha, get_fc_ha
    displacements = get_displacements(struc.get_positions(), 'geometry.in.supercell')
    F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
    E_ha = get_E_ha(displacements, F_ha)
    E_step = np.average(energies)

    threshold = force_norm_std[uncert_max_idx] - criteria.Un_Abs_F_avg_i + criteria.Un_Abs_F_std_i * 0.5
    if threshold >= 0:
        uncert_relat = threshold / (criteria.Un_Abs_F_std_i * 1.5)
        single_print(np.absolute(E_step - E_ha) / E_ha)
        single_print(np.absolute(E_step - E_ha))
        single_print(E_step)
        single_print(criteria.Epotential_avg)
        single_print(f'{np.absolute(E_step - E_ha) / E_ha > meta_Ediff}')
        single_print(f'{E_step >= criteria.Epotential_avg + 3 * criteria.Epotential_std}')

        if np.absolute(E_step - E_ha) / E_ha > meta_Ediff:
            single_print('deactivate')
            uncert_coeff[uncert_max_idx] = 0
        else:
            single_print('activate')
            uncert_coeff[uncert_max_idx] = min(uncert_relat, 1.0)

    F_ha_rev = np.array([coeff * np.array(force) for coeff, force in zip(uncert_coeff, F_ha)])
    return force_avg - F_ha_rev


def get_forces_temp(
    struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F,
    criteria, al_type, E_ref,
):
    force_avg, E_step_std, F_step_norm_std = _average_forces_and_step_std(
        struc, nstep, nmodel, calculator, E_ref, al_type,
    )
    return force_avg, E_step_std if al_type == 'energy_max' else F_step_norm_std


def get_forces_bias(
    struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F,
    criteria, bias_A, bias_B, idx_atom, E_ref,
):
    force_avg, _, _ = _bias_force_state(
        struc, nstep, nmodel, calculator, E_ref, bias_A, bias_B, idx_atom,
    )
    return force_avg


def get_forces_bias_temp(
    struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F,
    criteria, al_type, bias_A, bias_B, idx_atom, E_ref,
):
    force_avg, E_step_std, F_step_norm_std = _bias_force_state(
        struc, nstep, nmodel, calculator, E_ref, bias_A, bias_B, idx_atom,
    )
    return force_avg, E_step_std if al_type == 'energy_max' else F_step_norm_std


def _average_forces_and_step_std(struc, nstep, nmodel, calculator, E_ref, al_type):
    if not isinstance(calculator, list):
        struc.calc = calculator
        force_avg = struc.get_forces(md=True)
        return force_avg, np.zeros(len(struc)), np.zeros(len(struc))

    energies, forces = _collect_energies_and_forces(
        struc, nstep, nmodel, calculator, E_ref, al_type,
    )
    force_avg = np.average(forces, axis=0)
    return force_avg, np.std(energies, axis=0), _force_norm_std(forces, force_avg)


def _bias_force_state(struc, nstep, nmodel, calculator, E_ref, bias_A, bias_B, idx_atom):
    if not isinstance(calculator, list):
        struc.calc = calculator
        force_avg = struc.get_forces(md=True)
        return force_avg, np.zeros(len(struc)), np.zeros(len(struc))

    energies, forces = _collect_atomic_energies_and_forces(
        struc, nstep, nmodel, calculator, E_ref,
    )
    E_step_avg = np.average(energies, axis=0)
    E_step_std = np.std(energies, axis=0)
    F_step_avg = np.average(forces, axis=0)
    F_step_norm_std = _force_norm_std(forces, F_step_avg)

    E_bias_deriv = (
        E_step_std[idx_atom] / (bias_B ** 2) * bias_A
        * np.exp((-1) * (E_step_std[idx_atom] ** 2) / (2 * bias_B ** 2))
    )
    F_bias_elem = np.zeros(3)
    for idx_E, idx_F in zip(energies[:, idx_atom], forces[:, idx_atom]):
        F_bias_elem += (idx_E - E_step_avg[idx_atom]) * (idx_F - F_step_avg[idx_atom])

    single_print(f'idx_atom:{idx_atom}| E_bias_deriv:{E_bias_deriv}, F_bias_elem:{F_bias_elem}')
    single_print(f'Uncert_E:{E_step_std[idx_atom]}')
    F_bias = E_bias_deriv * F_bias_elem

    norm_F_step = np.linalg.norm(F_step_avg[idx_atom])
    norm_F_bias = np.linalg.norm(F_bias)
    ratio_bias = norm_F_bias / norm_F_step
    single_print(f'Bias Ratio: {np.average(ratio_bias)}\n')

    force_avg = F_step_avg.copy()
    force_avg[idx_atom] += F_bias / ratio_bias * 10 if ratio_bias > 10 else F_bias
    return force_avg, E_step_std, F_step_norm_std


def _collect_energies_and_forces(struc, nstep, nmodel, calculator, E_ref, al_type):
    if al_type == 'energy_max':
        return _collect_atomic_energies_and_forces(struc, nstep, nmodel, calculator, E_ref)
    return _collect_total_energies_and_forces(struc, nstep, nmodel, calculator, E_ref)


def _collect_total_energies_and_forces(struc, nstep, nmodel, calculator, E_ref):
    energies = []
    forces = []
    for idx in range(nstep * nmodel):
        struc.calc = calculator[idx]
        energies.append(struc.get_potential_energy() - E_ref[0][idx])
        forces.append(struc.get_forces())
    return np.asarray(energies), np.asarray(forces)


def _collect_atomic_energies_and_forces(struc, nstep, nmodel, calculator, E_ref):
    energies = []
    forces = []
    for idx in range(nstep * nmodel):
        struc.calc = calculator[idx]
        energies.append(struc.get_potential_energies() - E_ref[1][idx])
        forces.append(struc.get_forces())
    return np.asarray(energies), np.asarray(forces)


def _force_norm_std(forces, force_avg):
    force_norm = np.array([
        [np.linalg.norm(component) for component in model_forces]
        for model_forces in forces - force_avg
    ])
    return np.sqrt(np.average(force_norm ** 2, axis=0))


def _invalidate_cached_structure_data(struc):
    if hasattr(struc, 'update_structure'):
        struc.update_structure()
