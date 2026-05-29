import os
from decimal import Decimal

import ase.units as units
import numpy as np
from ase.io.trajectory import Trajectory, TrajectoryWriter

import torch

from libs.lib_criteria import eval_uncert, get_criteria, uncert_strconvter
from libs.lib_MD_util import get_forces, get_MDinfo_temp, get_masses

torch.set_default_dtype(torch.float64)


MODE_NORMAL = 'normal'
MODE_META = 'meta'


def NVTLangevin(
    struc, timestep, temperature, friction, steps, loginterval,
    nstep, nmodel, calculator, E_ref, al_type, trajectory, harmonic_F=False,
    anharmonic_F=False, logfile=None, signal_uncert=False,
    signal_append=True, fix_com=True,
):
    _run_nvtlangevin(
        MODE_NORMAL, struc, timestep, temperature, friction, steps,
        loginterval, nstep, nmodel, calculator, E_ref, al_type, trajectory,
        None, harmonic_F, anharmonic_F, logfile, signal_uncert,
        signal_append, fix_com,
    )


def NVTLangevin_meta(
    struc, timestep, temperature, friction, steps, loginterval,
    nstep, nmodel, calculator, E_ref, al_type, trajectory, meta_Ediff,
    harmonic_F=False, anharmonic_F=False, logfile=None,
    signal_uncert=False, signal_append=True, fix_com=True,
):
    _run_nvtlangevin(
        MODE_META, struc, timestep, temperature, friction, steps,
        loginterval, nstep, nmodel, calculator, E_ref, al_type, trajectory,
        meta_Ediff, harmonic_F, anharmonic_F, logfile, signal_uncert,
        signal_append, fix_com,
    )


def _run_nvtlangevin(
    mode, struc, timestep, temperature, friction, steps, loginterval,
    nstep, nmodel, calculator, E_ref, al_type, trajectory, meta_Ediff,
    harmonic_F, anharmonic_F, logfile, signal_uncert, signal_append,
    fix_com,
):
    langevin_idx, struc, file_traj = _open_trajectory(
        mode, trajectory, struc, loginterval, signal_append,
    )
    if langevin_idx == 0:
        _initialize_log(
            logfile, struc, nstep, nmodel, calculator, E_ref,
            al_type, harmonic_F, signal_uncert,
        )

    criteria = None
    if mode == MODE_META:
        criteria = get_criteria(temperature, 0, 0, 300, al_type)

    forces = _calculate_forces(
        mode, struc, nstep, nmodel, calculator, E_ref,
        harmonic_F, anharmonic_F, criteria, meta_Ediff,
    )

    for idx in range(langevin_idx, steps):
        natoms = len(struc)
        masses = get_masses(struc.get_masses(), natoms)
        c1, c2, c3, c4, c5 = _langevin_coefficients(
            timestep, temperature, friction, masses,
        )

        if forces is None:
            forces = _calculate_forces(
                mode, struc, nstep, nmodel, calculator, E_ref,
                harmonic_F, anharmonic_F, criteria, meta_Ediff,
            )

        velocity = struc.get_velocities()
        xi = np.random.standard_normal(size=(natoms, 3))
        eta = np.random.standard_normal(size=(natoms, 3))
        rnd_pos = c5 * eta
        rnd_vel = c3 * xi - c4 * eta

        if fix_com:
            rnd_pos -= rnd_pos.sum(axis=0) / natoms
            rnd_vel -= (rnd_vel * masses).sum(axis=0) / (masses * natoms)

        velocity += c1 * forces / masses - c2 * velocity + rnd_vel
        position = struc.get_positions()
        struc.set_positions(position + timestep * velocity + rnd_pos)
        velocity = (struc.get_positions() - position - rnd_pos) / timestep

        forces = _calculate_forces(
            mode, struc, nstep, nmodel, calculator, E_ref,
            harmonic_F, anharmonic_F, criteria, meta_Ediff,
        )
        velocity += c1 * forces / masses - c2 * velocity + rnd_vel
        struc.set_momenta(velocity * masses)

        if (idx + 1) % loginterval == 0:
            _append_step_log(
                logfile, struc, timestep, idx, loginterval, nstep,
                nmodel, calculator, E_ref, al_type, harmonic_F,
                signal_uncert,
            )
            file_traj.write(atoms=struc)

    file_traj.close()


def _open_trajectory(mode, trajectory, struc, loginterval, signal_append):
    if signal_append and os.path.exists(trajectory) and os.path.getsize(trajectory) != 0:
        traj_old = Trajectory(
            trajectory,
            properties=['forces', 'velocities', 'temperature'],
        )
        idx_factor = len(traj_old) if mode == MODE_META else len(traj_old) - 1
        langevin_idx = idx_factor * loginterval
        struc = traj_old[-1]
        del traj_old
        return langevin_idx, struc, TrajectoryWriter(filename=trajectory, mode='a')

    file_traj = TrajectoryWriter(filename=trajectory, mode='w')
    file_traj.write(atoms=struc)
    return 0, struc, file_traj


def _initialize_log(
    logfile, struc, nstep, nmodel, calculator, E_ref,
    al_type, harmonic_F, signal_uncert,
):
    if not isinstance(logfile, str):
        return

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
        struc, nstep, nmodel, calculator, harmonic_F, E_ref,
    )
    if signal_uncert:
        uncerts, _, S_step = eval_uncert(
            struc, nstep, nmodel, E_ref, calculator, al_type, harmonic_F,
        )

    with open(logfile, 'a') as file_log:
        file_log.write(_format_md_log_line(0.0, info_TE, info_PE, info_KE, info_T))
        if signal_uncert:
            file_log.write(_format_uncert_log_suffix(uncerts, S_step))
        else:
            file_log.write('\n')


def _append_step_log(
    logfile, struc, timestep, idx, loginterval, nstep, nmodel,
    calculator, E_ref, al_type, harmonic_F, signal_uncert,
):
    if not isinstance(logfile, str):
        return

    info_TE, info_PE, info_KE, info_T = get_MDinfo_temp(
        struc, nstep, nmodel, calculator, harmonic_F, E_ref,
    )
    if signal_uncert:
        uncerts, _, S_step = eval_uncert(
            struc, nstep, nmodel, E_ref, calculator, al_type, harmonic_F,
        )

    simtime = timestep * (idx + loginterval) / units.fs / 1000
    with open(logfile, 'a') as file_log:
        file_log.write(_format_md_log_line(simtime, info_TE, info_PE, info_KE, info_T))
        if signal_uncert:
            file_log.write(_format_uncert_log_suffix(uncerts, S_step))
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


def _format_uncert_log_suffix(uncerts, S_step):
    return (
        '      \t'
        + uncert_strconvter(uncerts.UncertAbs_E) + '\t'
        + uncert_strconvter(uncerts.UncertRel_E) + '\t'
        + uncert_strconvter(uncerts.UncertAbs_F) + '\t'
        + uncert_strconvter(uncerts.UncertRel_F) + '\t'
        + uncert_strconvter(uncerts.UncertAbs_S) + '\t'
        + uncert_strconvter(uncerts.UncertRel_S) + '\t'
        + uncert_strconvter(S_step) + '\n'
    )


def _calculate_forces(
    mode, struc, nstep, nmodel, calculator, E_ref,
    harmonic_F, anharmonic_F, criteria, meta_Ediff,
):
    if mode == MODE_META:
        from libs.lib_cont_nvtlangevin import get_forces_meta
        return get_forces_meta(
            struc, nstep, nmodel, calculator, harmonic_F,
            anharmonic_F, criteria, meta_Ediff, E_ref,
        )

    try:
        return struc.get_forces()
    except Exception:
        return get_forces(
            struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F,
        )


def _langevin_coefficients(timestep, temperature, friction, masses):
    sigma = np.sqrt(2 * temperature * friction / masses)
    c1 = timestep / 2.0 - timestep * timestep * friction / 8.0
    c2 = timestep * friction / 2.0 - timestep * timestep * friction * friction / 8.0
    c3 = np.sqrt(timestep) * sigma / 2.0 - timestep**1.5 * friction * sigma / 8.0
    c5 = timestep**1.5 * sigma / (2 * np.sqrt(3))
    c4 = friction / 2.0 * c5
    return c1, c2, c3, c4, c5
