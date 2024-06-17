from ase.io.trajectory import Trajectory
from ase.io.trajectory import TrajectoryWriter
from ase.io.bundletrajectory import BundleTrajectory
import ase.units as units
from ase.io.cif        import write_cif

import time
import os
import random
import numpy as np
from decimal import Decimal

from libs.lib_util    import single_print
from libs.lib_MD_util import get_forces, get_stress, get_MDinfo_temp, get_masses
from libs.lib_criteria import eval_uncert, uncert_strconvter, get_criteria, get_criteria_prob

import torch
torch.set_default_dtype(torch.float64)

def cont_NPTisoiso(
    inputs, struc, timestep, temperature, pressure, ttime, calculator, E_ref,
    MD_index, MD_step_index, signal_uncert=False, signal_append=True
):

    def traj_extra_contents_init():
        return np.array([timestep, temperature, desiredEkin, pressure, ttime, tfact, inputs.pfactor, pfact, frac_traceless])

    def traj_extra_contents():
        return np.array([zeta, zeta_past, zeta_integrated, MD_step_index])

    def traj_extra_contents_eta():
        return np.array([eta, eta_past])

    def traj_extra_contents_cell():
        return np.array([cell, cell_past])

    # Initialization of index
    condition = f'{inputs.temperature}K-{inputs.pressure}bar'

    trajectory = f'TEMPORARY/temp-{condition}_{inputs.index}.bundle'
    logfile = f'TEMPORARY/temp-{condition}_{inputs.index}.log'

    # Extract the criteria information from the initialization step
    criteria_collected = get_criteria(inputs.temperature, inputs.pressure, inputs.index, inputs.steps_init, inputs.al_type)

    if os.path.exists(trajectory):
        traj_temp = BundleTrajectory(filename=trajectory, mode='r')
        struc = traj_temp[-1]
        MD_step_index = len(traj_temp)
        del traj_temp

    # mpi_print(f'Step 1: {time.time()-time_init}', rank)
    if MD_step_index == 0: # If appending and the file exists,
        file_traj = BundleTrajectory(filename=trajectory, mode='w')
        # Add new configuration to the trajectory file
            
        if isinstance(logfile, str):
            file_log = open(logfile, 'w')
            file_log.write(
                'Time[ps]   \tEtot[eV]   \tEpot[eV]    \tEkin[eV]   \t'
                + 'Temperature[K]\t' + 'Pressure[GPa]'
                )
            if signal_uncert:
                file_log.write(
                    '\tUncertAbs_E\tUncertRel_E\t'
                    + 'UncertAbs_F\tUncertRel_F\t'
                    + 'UncertAbs_S\tUncertRel_S\tS_average\n'
                    )
            else:
                file_log.write('\n')
            file_log.close()
        
            # Get MD information at the current step
            info_TE, info_PE, info_KE, info_T, info_P = get_MDinfo_temp(
                struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F, E_ref, signal_P = True
                )

            if signal_uncert:
                # Get absolute and relative uncertainties of energy and force
                # and also total energy
                uncerts, Epot_step, S_step =\
                eval_uncert(struc, inputs.nstep, inputs.nmodel, E_ref, calculator, inputs.al_type, inputs.harmonic_F)

            # Log MD information at the current step in the log file
            file_log = open(logfile, 'a')
            file_log.write(
                '{:.5f}'.format(Decimal(str(0.0))) + '   \t' +
                '{:.5e}'.format(Decimal(str(info_TE))) + '\t' +
                '{:.5e}'.format(Decimal(str(info_PE))) + '\t' +
                '{:.5e}'.format(Decimal(str(info_KE))) + '\t' +
                '{:.2f}'.format(Decimal(str(info_T))) + '\t' +
                '        ' + '{:.5e}'.format(Decimal(str(info_P)))
                )
            if signal_uncert:
                file_log.write(
                    '\t' +
                    uncert_strconvter(uncerts.UncertAbs_E) + '\t' +
                    uncert_strconvter(uncerts.UncertRel_E) + '\t' +
                    uncert_strconvter(uncerts.UncertAbs_F) + '\t' +
                    uncert_strconvter(uncerts.UncertRel_F) + '\t' +
                    uncert_strconvter(uncerts.UncertAbs_S) + '\t' +
                    uncert_strconvter(uncerts.UncertRel_S) + '\n'
                    )
            else:
                file_log.write('\n')
            file_log.close()

        NumAtoms = len(struc)
        struc = zero_center_of_mass_momentum(struc, NumAtoms, rank, verbose=1)
        eta = np.zeros((3, 3), float)
        zeta = 0.0
        zeta_integrated = 0.0
        initialized = 0
        cell = struc.get_cell()
        tfact, pfact, desiredEkin = calculateconstants(ttime, temperature, inputs.pfactor, NumAtoms, cell)
        externalstress = np.array([-pressure,-pressure,-pressure, 0.0, 0.0, 0.0])
        frac_traceless = 1

        if not cell[1, 0] == cell[2, 0] == cell[2, 1] == 0.0:
            single_print("cell:")
            single_print(cell)
            single_print("Min:", min((cell[1, 0], cell[2, 0], cell[2, 1])))
            single_print("Max:", max((cell[1, 0], cell[2, 0], cell[2, 1])))
            raise NotImplementedError(
                "Can (so far) only operate on lists of atoms where the "
                "computational box is an upper triangular matrix.")

        inv_cell = np.linalg.inv(cell)
        q = np.dot(struc.get_positions(), inv_cell) - 0.5

        # Get averaged force from trained models
        forces = get_forces(struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F, inputs.anharmonic_F)

        # Get averaged stress from trained models
        stress = get_stress(struc, inputs.nstep, inputs.nmodel, calculator)

        cell_past, eta_past = initialize_eta_h(cell, timestep, eta, inputs.pfactor, pfact, stress, frac_traceless, inputs.mask)
        deltazeta = timestep * tfact * (struc.get_kinetic_energy() - desiredEkin)
        zeta_past = zeta - deltazeta

        q_past, q_future = calculate_q_past_and_future(struc, timestep, cell, inv_cell, forces, eta, zeta, q)

        initialized = 1

        file_traj.set_extra_data('npt_init', traj_extra_contents_init,  once=True)
        file_traj.set_extra_data('npt_dyn', traj_extra_contents)
        file_traj.set_extra_data('npt_dyn_eta', traj_extra_contents_eta)
        file_traj.set_extra_data('npt_dyn_cell', traj_extra_contents_cell)
        file_traj.write(atoms=struc)
    else:
        file_traj_read = BundleTrajectory(filename=trajectory, mode='r')
        file_traj_read[0]; #ASE bug

        inputs_npt_init = file_traj_read.read_extra_data('npt_init', -1)
        desiredEkin = inputs_npt_init[2]
        pressure = inputs_npt_init[3]
        externalstress = np.array([-pressure,-pressure,-pressure, 0.0, 0.0, 0.0])
        ttime = inputs_npt_init[4]
        tfact = inputs_npt_init[5]
        pfact = inputs_npt_init[7]
        frac_traceless = inputs_npt_init[8]

        # Get averaged force from trained models
        try:
            forces = struc.get_forces()
        except Exception as e:
            forces = get_forces(struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F, inputs.anharmonic_F)

        # Get averaged stress from trained models
        try:
            stress = struc.get_stress(include_ideal_gas=True)
        except Exception as e:
            stress = get_stress(struc, inputs.nstep, inputs.nmodel, calculator)

        inputs_npt_dyn = file_traj_read.read_extra_data('npt_dyn', -1)
        zeta = inputs_npt_dyn[0]
        zeta_past = inputs_npt_dyn[1]
        zeta_integrated = inputs_npt_dyn[2]
        MD_step_index = int(inputs_npt_dyn[3])

        inputs_npt_dyn_eta = file_traj_read.read_extra_data('npt_dyn_eta', -1)
        eta = inputs_npt_dyn_eta[0]
        eta_past = inputs_npt_dyn_eta[1]

        inputs_npt_dyn_cell = file_traj_read.read_extra_data('npt_dyn_cell', -1)
        cell = inputs_npt_dyn_cell[0]
        cell_past = inputs_npt_dyn_cell[1]

        del file_traj_read

        NumAtoms = len(struc)
        cell = struc.get_cell()

        if not cell[1, 0] == cell[2, 0] == cell[2, 1] == 0.0:
            single_print("cell:")
            single_print(cell)
            single_print("Min:", min((cell[1, 0], cell[2, 0], cell[2, 1])))
            single_print("Max:", max((cell[1, 0], cell[2, 0], cell[2, 1])))
            raise NotImplementedError(
                "Can (so far) only operate on lists of atoms where the "
                "computational box is an upper triangular matrix.")

        inv_cell = np.linalg.inv(cell)
        q = np.dot(struc.get_positions(), inv_cell) - 0.5

        deltazeta = timestep * tfact * (struc.get_kinetic_energy() - desiredEkin)

        q_past, q_future = calculate_q_past_and_future(struc, timestep, cell, inv_cell, forces, eta, zeta, q)

        initialized = 1

        file_traj = BundleTrajectory(filename=trajectory, mode='a')
        file_traj.set_extra_data('npt_dyn', traj_extra_contents)
        file_traj.set_extra_data('npt_dyn_eta', traj_extra_contents_eta)
        file_traj.set_extra_data('npt_dyn_cell', traj_extra_contents_cell)

    write_traj = TrajectoryWriter(
        filename=f'TRAJ/traj-{condition}_{inputs.index+1}.traj',
        mode='a'
        )

    # Go trough steps until the requested number of steps
    # If appending, it starts from Langevin_idx. Otherwise, Langevin_idx = 0
    while (MD_index < inputs.ntotal) or (inputs.calc_type == 'period' and MD_step_index < inputs.nperiod*inputs.loginterval):
        cell_future = cell_past + 2 * timestep * np.dot(cell, eta)

        if inputs.pfactor is None:
            deltaeta = np.zeros(6, float)
        else:
            deltaeta = -2 * timestep * pfact * np.linalg.det(cell) * (stress - externalstress)

        if frac_traceless == 1:
            eta_future = eta_past + inputs.mask * makeuppertriangular(deltaeta)
        else:
            trace_part, traceless_part = separatetrace(makeuppertriangular(deltaeta))
            eta_future = eta + trace_part + frac_traceless * tracelsss_part

        deltazeta = 2 * timestep * tfact * (struc.get_kinetic_energy() - desiredEkin)
        zeta_future = zeta_past + deltazeta

        # Advance time
        cell_past = cell
        cell = cell_future
        inv_cell = np.linalg.inv(cell)
        q_past = q
        q = q_future

        struc.set_cell(cell)
        r = np.dot(q + 0.5, cell)
        struc.set_positions(r)

        eta_past = eta
        eta = eta_future
        zeta_past = zeta
        zeta = zeta_future
        zeta_integrated += timestep * zeta

        forces = get_forces(struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F, inputs.anharmonic_F)
        q_future = calculate_q_future(struc, timestep, cell, inv_cell, forces, eta, zeta, q, q_past)
        struc.set_momenta(np.dot(q_future - q_past, cell / (2 * timestep)) * np.reshape(struc.get_masses(), (-1, 1)))
        stress = get_stress(struc, inputs.nstep, inputs.nmodel, calculator)

        # mpi_print(f'Step 11: {time.time()-time_init}', rank)
        # Log MD information at regular intervals
        if (MD_step_index+1) % inputs.loginterval == 0:

            # Get absolute and relative uncertainties of energy and force
            # and also total energy
            uncerts, Epot_step, S_step =\
            eval_uncert(struc, inputs.nstep, inputs.nmodel, E_ref, calculator, inputs.al_type, inputs.harmonic_F)

            # Get a criteria probability from uncertainty and energy informations
            criteria = get_criteria_prob(inputs, Epot_step, uncerts, criteria_collected)

            # mpi_print(f'Step 12: {time.time()-time_init}', rank)
            info_TE, info_PE, info_KE, info_T, info_P = get_MDinfo_temp(
                struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F, E_ref, signal_P = True
                )

            # Acceptance check with criteria
            ##!! Epot_step should be rechecked.
            if random.random() < criteria: # and Epot_step > 0.1:
                accept = 'Accepted'
                MD_index += 1
                write_traj.write(atoms=struc)
            else:
                accept = 'Vetoed'

            # Record the MD results at the current step
            trajfile = open(f'UNCERT/uncertainty-{condition}_{inputs.index}.txt', 'a')
            trajfile.write(
                '{:.5e}'.format(Decimal(str(info_T))) + '\t' +
                '{:.5e}'.format(Decimal(str(info_P))) + '\t' +
                uncert_strconvter(uncerts.UncertAbs_E) + '\t' +
                uncert_strconvter(uncerts.UncertRel_E) + '\t' +
                uncert_strconvter(uncerts.UncertAbs_F) + '\t' +
                uncert_strconvter(uncerts.UncertRel_F) + '\t' +
                uncert_strconvter(uncerts.UncertAbs_S) + '\t' +
                uncert_strconvter(uncerts.UncertRel_S) + '\t' +
                uncert_strconvter(Epot_step) + '\t' +
                uncert_strconvter(S_step) + '\t' +
                str(MD_index) + '          \t' +
                '{:.5e}'.format(Decimal(str(criteria))) + '\t' +
                str(accept) + '   \n'
            )
            trajfile.close()

            # mpi_print(f'Step 14: {time.time()-time_init}', rank)
            file_log = open(logfile, 'a')
            simtime = timestep*MD_step_index/units.fs/1000
            file_log.write(
                '{:.5f}'.format(Decimal(str(simtime))) + '   \t' +
                '{:.5e}'.format(Decimal(str(info_TE))) + '\t' +
                '{:.5e}'.format(Decimal(str(info_PE))) + '\t' +
                '{:.5e}'.format(Decimal(str(info_KE))) + '\t' +
                '{:.2f}'.format(Decimal(str(info_T))) + '\t' +
                '        ' + '{:.5e}'.format(Decimal(str(info_P)))
                )
            if signal_uncert:
                file_log.write(
                    '\t' +
                    uncert_strconvter(uncerts.UncertAbs_E) + '\t' +
                    uncert_strconvter(uncerts.UncertRel_E) + '\t' +
                    uncert_strconvter(uncerts.UncertAbs_F) + '\t' +
                    uncert_strconvter(uncerts.UncertRel_F) + '\t' +
                    uncert_strconvter(uncerts.UncertAbs_S) + '\t' +
                    uncert_strconvter(uncerts.UncertRel_S) + '\n'
                    )
            else:
                file_log.write('\n')
            file_log.close()
            # mpi_print(f'Step 15: {time.time()-time_init}', rank)
            file_traj.write(atoms=struc)

        MD_step_index += 1
        # mpi_print(f'Step 16: {time.time()-time_init}', rank)


def calculateconstants(
    ttime, temperature, pfactor, NumAtoms, cell
):
    tfact = 2.0 / (3 * NumAtoms * temperature * ttime * ttime)

    if pfactor is None:
        pfact = 0.0
    else:
        pfact = 1.0 / (pfactor * np.linalg.det(cell))

    desiredEkin = 1.5 * (NumAtoms - 1) * temperature

    return tfact, pfact, desiredEkin


def initialize_eta_h(
    cell, timestep, eta, pfactor, pfact, stress, frac_traceless, mask
):
    cell_past = cell - timestep * np.dot(cell, eta)

    if pfactor is None:
        deltaeta = np.zeros(6, float)
    else:
        deltaeta = (-timestep) * pfact * np.linalg.det(cell) * stress
    
    if frac_traceless == 1:
        eta_past = eta - mask * makeuppertriangular(deltaeta)
    else:
        trace_part, traceless_part = separatetrace(makeuppertriangular(deltaeta))
        eta_past = eta - trace_part - frac_traceless * tracelsss_part

    return cell_past, eta_past


def makeuppertriangular(sixvector):
    """Make an upper triangular matrix from a 6-vector."""
    return np.array(((sixvector[0], sixvector[5], sixvector[4]),
                     (0, sixvector[1], sixvector[3]),
                     (0, 0, sixvector[2])))


def separatetrace(mat):
    """return two matrices, one proportional to the identity
    the other traceless, which sum to the given matrix
    """
    tracePart = ((mat[0][0] + mat[1][1] + mat[2][2]) / 3.) * np.identity(3)
    return tracePart, mat - tracePart


def calculate_q_future(struc, timestep, cell, inv_cell, forces, eta, zeta, q, q_past):
    id3 = np.identity(3)
    alpha = (timestep * timestep) * np.dot(forces / np.reshape(struc.get_masses(), (-1, 1)), inv_cell)
    beta = timestep * np.dot(cell, np.dot(eta + 0.5 * zeta * id3, inv_cell))
    inv_b = np.linalg.inv(beta + id3)
    q_future = np.dot(2 * q + np.dot(q_past, beta - id3) + alpha, inv_b)

    return q_future


def calculate_q_past_and_future(struc, timestep, cell, inv_cell, forces, eta, zeta, q):
    def ekin(p, m=np.reshape(struc.get_masses(), (-1, 1))):
        p2 = np.sum(p * p, -1)
        return 0.5 * np.sum(p2 / m) / len(m)
    p0 = struc.get_momenta()
    m = np.reshape(struc.get_masses(), (-1, 1))
    p = np.array(p0, copy=1)
    
    for i in range(2):
        q_past = q - timestep * np.dot(p / m, inv_cell)
        q_future = calculate_q_future(struc, timestep, cell, inv_cell, forces, eta, zeta, q, q_past)

        p = np.dot(q_future - q_past, cell / (2 * timestep)) * m
        e = ekin(p)
        if e < 1e-5:
            # The kinetic energy and momenta are virtually zero
            return q_past, q_future
        p = (p0 - p) + p0

    return q_past, q_future


def zero_center_of_mass_momentum(struc, NumAtoms, rank, verbose=0):
    "Set the center of mass momentum to zero."
    cm = struc.get_momenta().sum(0)
    abscm = np.sqrt(np.sum(cm * cm))
    if verbose and abscm > 1e-4:
        single_print(
            "Setting the center-of-mass momentum to zero "
            "(was %.6g %.6g %.6g)" % tuple(cm)
            )
    struc.set_momenta(struc.get_momenta() - cm / NumAtoms)
    return struc
