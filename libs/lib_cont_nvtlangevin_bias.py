from ase.io.trajectory import Trajectory
from ase.io.trajectory import TrajectoryWriter
import ase.units as units
from ase.io.cif        import write_cif

import time
import os
import random
import numpy as np
import pandas as pd
from mpi4py import MPI
from decimal import Decimal
from ase.build import make_supercell
from ase.io import read as atoms_read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from libs.lib_util    import single_print, mpi_print
from libs.lib_MD_util import get_forces, get_MDinfo_temp, get_masses
from libs.lib_criteria import eval_uncert, uncert_strconvter, get_criteria, get_criteria_prob

import torch
torch.set_default_dtype(torch.float64)

def cont_NVTLangevin_bias(
    inputs, struc, timestep, temperature, calculator, E_ref,
    MD_index, MD_step_index, signal_uncert=False, signal_append=True, fix_com=True,
):
    """Function [NVTLangevin]
    Evalulate the absolute and relative uncertainties of
    predicted energies and forces.
    This script is adopted from ASE Langevin function 
    and modified to use averaged forces from trained model.

    Parameters:

    struc: ASE atoms
        A structral configuration of a starting point
    timestep: float
        The step interval for printing MD steps
    temperature: float
        The desired temperature in units of Kelvin (K)

    friction: float
        Strength of the friction parameter in NVTLangevin ensemble
    steps: int
        The length of the Molecular Dynamics steps
    loginterval: int
        The step interval for printing MD steps

    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization
    calculator: ASE calculator
        Any calculator
    E_ref: flaot
        The energy of reference state (Here, ground state)
    al_type: str
        Type of active learning: 'energy', 'force', 'force_max'
    trajectory: str
        A name of MD trajectory file

    logfile: str (optional)
        A name of MD logfile. With None, it will not print a log file.
    signal_uncert: bool (optional)


    fixcm: bool (optional)
        If True, the position and momentum of the center of mass is
        kept unperturbed.  Default: True.
    """

    time_init = time.time()

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialization of index
    condition = f'{inputs.temperature}K-{inputs.pressure}bar'

    trajectory = f'TEMPORARY/temp-{condition}_{inputs.index}.traj'
    logfile = f'TEMPORARY/temp-{condition}_{inputs.index}.log'

    # Extract the criteria information from the initialization step
    criteria_collected = get_criteria(inputs.temperature, inputs.pressure, inputs.index, inputs.steps_init, inputs.al_type)

    if os.path.exists(trajectory):
        traj_temp = Trajectory(trajectory)
        struc = traj_temp[-1]
        MD_step_index = len(traj_temp)
        del traj_temp

    # mpi_print(f'Step 1: {time.time()-time_init}', rank)
    if MD_step_index == 0: # If appending and the file exists,
        file_traj = TrajectoryWriter(filename=trajectory, mode='w')
        # Add new configuration to the trajectory file
        if rank == 0:
            file_traj.write(atoms=struc)
            
        if isinstance(logfile, str):
            if rank == 0:
                file_log = open(logfile, 'w')
                file_log.write(
                    'Time[ps]   \tEtot[eV]   \tEpot[eV]    \tEkin[eV]   \t'
                    + 'Temperature[K]'
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
            info_TE, info_PE, info_KE, info_T = get_MDinfo_temp(
                struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F
                )

            if signal_uncert:
                # Get absolute and relative uncertainties of energy and force
                # and also total energy
                uncerts, Epot_step, S_step =\
                eval_uncert(struc, inputs.nstep, inputs.nmodel, E_ref, calculator, inputs.al_type, inputs.harmonic_F)

            # Log MD information at the current step in the log file
            if rank == 0:
                file_log = open(logfile, 'a')
                file_log.write(
                    '{:.5f}'.format(Decimal(str(0.0))) + '   \t' +
                    '{:.5e}'.format(Decimal(str(info_TE))) + '\t' +
                    '{:.5e}'.format(Decimal(str(info_PE))) + '\t' +
                    '{:.5e}'.format(Decimal(str(info_KE))) + '\t' +
                    '{:.2f}'.format(Decimal(str(info_T)))
                    )
                if signal_uncert:
                    file_log.write(
                        '      \t' +
                        uncert_strconvter(uncerts.UncertAbs_E) + '\t' +
                        uncert_strconvter(uncerts.UncertRel_E) + '\t' +
                        uncert_strconvter(uncerts.UncertAbs_F) + '\t' +
                        uncert_strconvter(uncerts.UncertRel_F) + '\t' +
                        uncert_strconvter(uncerts.UncertAbs_S) + '\t' +
                        uncert_strconvter(uncerts.UncertRel_S) + '\t' +
                        uncert_strconvter(S_step) + '\n'
                        )
                else:
                    file_log.write('\n')
                file_log.close()
    else:
        file_traj = TrajectoryWriter(filename=trajectory, mode='a')

    write_traj = TrajectoryWriter(
        filename=f'TRAJ/traj-{condition}_{inputs.index+1}.traj',
        mode='a'
        )

    # Get averaged force from trained models
    forces = get_forces_bias(struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F, inputs.anharmonic_F, criteria_collected, inputs.bias_A, inputs.bias_B, inputs.idx_atom)

    # mpi_print(f'Step 3: {time.time()-time_init}', rank)
    # Go trough steps until the requested number of steps
    # If appending, it starts from Langevin_idx. Otherwise, Langevin_idx = 0
    while (MD_index < inputs.ntotal) or (inputs.calc_type == 'period' and MD_step_index < inputs.nperiod*inputs.loginterval):

        accept = '--         '
        natoms = len(struc)

        # mpi_print(f'Step 6: {time.time()-time_init}', rank)
        # Get averaged forces and velocities
        if forces is None:
            forces = get_forces_bias(struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F, inputs.anharmonic_F, criteria_collected, inputs.bias_A, inputs.bias_B, inputs.idx_atom)
        # Velocity is already calculated based on averaged forces
        # in the previous step
        velocity = struc.get_velocities()
        
        # mpi_print(f'Step 7: {time.time()-time_init}', rank)
        # Sample the random numbers for the temperature fluctuation
        xi = np.empty(shape=(natoms, 3))
        eta = np.empty(shape=(natoms, 3))
        if rank == 0:
            xi = np.random.standard_normal(size=(natoms, 3))
            eta = np.random.standard_normal(size=(natoms, 3))
        comm.Bcast(xi, root=0)
        comm.Bcast(eta, root=0)

        # mpi_print(f'Step 4: {time.time()-time_init}', rank)
        # Get essential properties
        masses = get_masses(struc.get_masses(), natoms)

        # mpi_print(f'Step 5: {time.time()-time_init}', rank)
        # Get Langevin coefficients
        sigma = np.sqrt(2 * temperature * inputs.friction / masses)
        c1 = timestep / 2. - timestep * timestep * inputs.friction / 8.
        c2 = timestep * inputs.friction / 2 - timestep * timestep * inputs.friction * inputs.friction / 8.
        c3 = np.sqrt(timestep) * sigma / 2. - timestep**1.5 * inputs.friction * sigma / 8.
        c5 = timestep**1.5 * sigma / (2 * np.sqrt(3))
        c4 = inputs.friction / 2. * c5

        # mpi_print(f'Step 8: {time.time()-time_init}', rank)
        # Get get changes of positions and velocities
        rnd_pos = c5 * eta
        rnd_vel = c3 * xi - c4 * eta
        
        # Check the center of mass
        if fix_com:
            rnd_pos -= rnd_pos.sum(axis=0) / natoms
            rnd_vel -= (rnd_vel * masses).sum(axis=0) / (masses * natoms)
            
        # First halfstep in the velocity.
        velocity += (c1 * forces / masses - c2 * velocity + rnd_vel)
        
        # mpi_print(f'Step 9: {time.time()-time_init}', rank)
        # Full step in positions
        position = struc.get_positions()
        
        # Step: x^n -> x^(n+1) - this applies constraints if any.
        struc.set_positions(position + timestep * velocity + rnd_pos)

        # mpi_print(f'Step 10: {time.time()-time_init}', rank)
        # recalc velocities after RATTLE constraints are applied
        velocity = (struc.get_positions() - position - rnd_pos) / timestep
        comm.Barrier()
        # mpi_print(f'Step 10-1: {time.time()-time_init}', rank)
        forces = get_forces_bias(struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F, inputs.anharmonic_F, criteria_collected, inputs.bias_A, inputs.bias_B, inputs.idx_atom)
        comm.Barrier()
        
        # mpi_print(f'Step 10-2: {time.time()-time_init}', rank)
        # Update the velocities
        velocity += (c1 * forces / masses - c2 * velocity + rnd_vel)

        # mpi_print(f'Step 10-3: {time.time()-time_init}', rank)
        # Second part of RATTLE taken care of here
        struc.set_momenta(velocity * masses)
        
        # mpi_print(f'Step 11: {time.time()-time_init}', rank)
        # Log MD information at regular intervals
        if (MD_step_index+1) % inputs.loginterval == 0:

            # Get absolute and relative uncertainties of energy and force
            # and also total energy
            uncerts, Epot_step, S_step =\
            eval_uncert(struc, inputs.nstep, inputs.nmodel, 0.0, calculator, inputs.al_type, inputs.harmonic_F)

            # Get a criteria probability from uncertainty and energy informations
            criteria = get_criteria_prob(inputs, Epot_step, uncerts, criteria_collected)

            if inputs.rank == 0:
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
                    '{:.5e}'.format(Decimal(str(struc.get_temperature()))) + '\t' +
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

            if isinstance(logfile, str):
                # mpi_print(f'Step 12: {time.time()-time_init}', rank)
                info_TE, info_PE, info_KE, info_T = get_MDinfo_temp(
                    struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F
                    )

                # mpi_print(f'Step 14: {time.time()-time_init}', rank)
                if inputs.rank == 0:
                    file_log = open(logfile, 'a')
                    simtime = timestep*(MD_step_index+inputs.loginterval)/units.fs/1000
                    file_log.write(
                        '{:.5f}'.format(Decimal(str(simtime))) + '   \t' +
                        '{:.5e}'.format(Decimal(str(info_TE))) + '\t' +
                        '{:.5e}'.format(Decimal(str(info_PE))) + '\t' +
                        '{:.5e}'.format(Decimal(str(info_KE))) + '\t' +
                        '{:.2f}'.format(Decimal(str(info_T)))
                        )
                    if signal_uncert:
                        file_log.write(
                            '      \t' +
                            uncert_strconvter(uncerts.UncertAbs_E) + '\t' +
                            uncert_strconvter(uncerts.UncertRel_E) + '\t' +
                            uncert_strconvter(uncerts.UncertAbs_F) + '\t' +
                            uncert_strconvter(uncerts.UncertRel_F) + '\t' +
                            uncert_strconvter(uncerts.UncertAbs_S) + '\t' +
                            uncert_strconvter(uncerts.UncertRel_S) + '\t' +
                            uncert_strconvter(S_step) + '\n'
                            )
                    else:
                        file_log.write('\n')
                    file_log.close()
                # mpi_print(f'Step 15: {time.time()-time_init}', rank)
            if rank == 0:
                file_traj.write(atoms=struc)
            MD_index = inputs.comm.bcast(MD_index, root=0)

        MD_step_index += 1
        MD_step_index = inputs.comm.bcast(MD_step_index, root=0)
        # mpi_print(f'Step 16: {time.time()-time_init}', rank)


def get_forces_bias(
    struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F, criteria, bias_A, bias_B, idx_atom
):
    """Function [get_forces]
    Evalulate the average of forces from all different trained models.

    Parameters:

    struc_step: ASE atoms
        A structral configuration at the current step
    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization
    calculator: ASE calculator or list of ASE calculators
        Calculators from trained models

    Returns:

    force_avg: np.array of float
        Averaged forces across trained models
    """

    # time_init = time.time()
    from libs.lib_util import eval_sigma

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # mpi_print(f'Step 10-a: {time.time()-time_init}', rank)
    if type(calculator) == list:
        energies = []
        forces = []
        sigmas = []
        zndex = 0
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                if (index_nmodel*nstep + index_nstep) % size == rank:
                    # mpi_print(f'Step 10-a1 first {rank}: {time.time()-time_init}', rank)
                    struc.calc = calculator[zndex]
                    # mpi_print(f'Step 10-a1 second {rank}: {time.time()-time_init}', rank)
                    temp_force = struc.get_forces()
                    # mpi_print(f'Step 10-a1 third {rank}: {time.time()-time_init}', rank)
                    energies.append(struc.get_potential_energies())
                    forces.append(temp_force)
                    # sigmas.append(eval_sigma(temp_force, struc.get_positions(), 'force_max'))
                    # mpi_print(f'Step 10-a1 last {rank}: {time.time()-time_init}', rank)
                    zndex += 1
        # mpi_print(f'Step 10-a2: {time.time()-time_init}', rank)
        energies = comm.allgather(energies)
        forces = comm.allgather(forces)
        # sigmas = comm.allgather(sigmas)

        E_step_filtered = [jtem for item in energies if len(item) != 0 for jtem in item]
        E_step_avg = np.average(E_step_filtered, axis=0)
        E_step_std = np.std(E_step_filtered, axis=0)

        F_step_filtered = [jtem for item in forces if len(item) != 0 for jtem in item]
        F_step_avg = np.average(F_step_filtered, axis=0)
        F_step_norm = np.array([[np.linalg.norm(Fcomp) for Fcomp in Ftems] for Ftems in F_step_filtered - F_step_avg])
        F_step_norm_std = np.sqrt(np.average(F_step_norm ** 2, axis=0))

        idx_E_std = np.argmax(F_step_norm_std)

        # !!!Entire!!!
        # F_bias = []
        # natoms = len(struc)
        # for idx in range(natoms): # mpi needs to be fixed. (coom. Sequence problem); Serial is fine.
        #     E_bias_deriv = E_step_std[idx] / (bias_B ** 2) * bias_A * (np.exp((-1) * (E_step_std[idx] ** 2) / (2 * bias_B ** 2)))
        #     F_bias_elem = np.array([0.0, 0.0, 0.0])
        #     for idx_E, idx_F in zip(np.array(E_step_filtered)[:,idx], np.array(F_step_filtered)[:,idx]):
        #         F_bias_elem += (idx_E - E_step_avg[idx])*(idx_F - F_step_avg[idx])
        #         # if idx == idx_E_std: print(f'idx_E:{idx_E - E_step_avg[idx]}, idx_F:{idx_F - F_step_avg[idx]}')
        #     if idx == idx_E_std: print(f'E_bias_deriv:{E_bias_deriv}, F_bias_elem:{F_bias_elem}')
        #     F_bias.append(E_bias_deriv * F_bias_elem)

        # force_avg = F_step_avg + np.array(F_bias)

        # norm_F_step = np.linalg.norm(F_step_avg, axis=1)
        # norm_F_bias = np.linalg.norm(F_bias, axis=1)

        # ratio_bias = []
        # for item_norm_F_step, item_norm_F_bias in zip(norm_F_step, norm_F_bias):
        #     ratio_bias.append(item_norm_F_bias/item_norm_F_step)
        # print(f'Bias Ratio: {np.average(ratio_bias)}\n')


        E_bias_deriv = E_step_std[idx_atom] / (bias_B ** 2) * bias_A * (np.exp((-1) * (E_step_std[idx_atom] ** 2) / (2 * bias_B ** 2)))
        F_bias_elem = np.array([0.0, 0.0, 0.0])
        for idx_E, idx_F in zip(np.array(E_step_filtered)[:,idx_atom], np.array(F_step_filtered)[:,idx_atom]):
            F_bias_elem += (idx_E - E_step_avg[idx_atom])*(idx_F - F_step_avg[idx_atom])
            # mpi_print(f'idx_E:{idx_E - E_step_avg[idx_atom]}, idx_F:{idx_F - F_step_avg[idx_atom]}', rank)
        mpi_print(f'idx_atom:{idx_atom}| E_bias_deriv:{E_bias_deriv}, F_bias_elem:{F_bias_elem}', rank)
        mpi_print(f'Uncert_E:{E_step_std[idx_atom]}', rank)
        F_bias = E_bias_deriv * F_bias_elem

        norm_F_step = np.linalg.norm(F_step_avg[idx_atom])
        norm_F_bias = np.linalg.norm(F_bias)

        ratio_bias = norm_F_bias/norm_F_step
        mpi_print(f'Bias Ratio: {np.average(ratio_bias)}\n', rank)

        force_avg = F_step_avg.copy()

        if ratio_bias > 10:
            F_bias = F_bias / ratio_bias * 10
            force_avg[idx_atom] += F_bias
        else:
            force_avg[idx_atom] += F_bias

    else:
        forces = None
        if rank == 0:
            struc.calc = calculator
            forces = struc.get_forces(md=True)
        force_avg = comm.bcast(forces, root=0)

    # mpi_print(f'Step 10-d: {time.time()-time_init}', rank)

    return force_avg
