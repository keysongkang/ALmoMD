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

def cont_NVTLangevin_meta(
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
    # try:
    #     forces = struc.get_forces()
    # except Exception as e:
    forces = get_forces_meta(struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F, inputs.anharmonic_F, criteria_collected, inputs.meta_Ediff)

    # mpi_print(f'Step 3: {time.time()-time_init}', rank)
    # Go trough steps until the requested number of steps
    # If appending, it starts from Langevin_idx. Otherwise, Langevin_idx = 0
    while (MD_index < inputs.ntotal) or (inputs.calc_type == 'period' and MD_step_index < inputs.nperiod*inputs.loginterval):

        if inputs.meta_restart == True:
            if os.path.exists('start.in') and inputs.ensemble == 'NVTLangevin_meta' and MD_index != 0:
                uncert_file = f'UNCERT/uncertainty-{condition}_{inputs.index}.txt'
                uncert_data = pd.read_csv(uncert_file, index_col=False, delimiter='\t')

                if np.array(uncert_data.loc[:,'S_average'])[-1] > inputs.meta_r_crtria:
                    mpi_print(f'[MLMD] Read a configuration from start.in', inputs.rank)
                    # Read the ground state structure with the primitive cell
                    struc_init = atoms_read('start.in', format='aims')
                    # Make it supercell
                    struc = make_supercell(struc_init, inputs.supercell_init)
                    MaxwellBoltzmannDistribution(struc, temperature_K=inputs.temperature*1.5, force_temp=True)

        accept = '--         '

        # mpi_print(f'Step 4: {time.time()-time_init}', rank)
        # Get essential properties
        natoms = len(struc)
        masses = get_masses(struc.get_masses(), natoms)
        sigma = np.sqrt(2 * temperature * inputs.friction / masses)

        # mpi_print(f'Step 5: {time.time()-time_init}', rank)
        # Get Langevin coefficients
        c1 = timestep / 2. - timestep * timestep * inputs.friction / 8.
        c2 = timestep * inputs.friction / 2 - timestep * timestep * inputs.friction * inputs.friction / 8.
        c3 = np.sqrt(timestep) * sigma / 2. - timestep**1.5 * inputs.friction * sigma / 8.
        c5 = timestep**1.5 * sigma / (2 * np.sqrt(3))
        c4 = inputs.friction / 2. * c5

        # mpi_print(f'Step 6: {time.time()-time_init}', rank)
        # Get averaged forces and velocities
        if forces is None:
            forces = get_forces_meta(struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F, inputs.anharmonic_F, criteria_collected, inputs.meta_Ediff)
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
        forces = get_forces_meta(struc, inputs.nstep, inputs.nmodel, calculator, inputs.harmonic_F, inputs.anharmonic_F, criteria_collected, inputs.meta_Ediff)
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


def get_forces_meta(
    struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F, criteria, meta_Ediff
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
        energy = []
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
                    energy.append(struc.get_potential_energy())
                    forces.append(temp_force)
                    # sigmas.append(eval_sigma(temp_force, struc.get_positions(), 'force_max'))
                    # mpi_print(f'Step 10-a1 last {rank}: {time.time()-time_init}', rank)
                    zndex += 1
        # mpi_print(f'Step 10-a2: {time.time()-time_init}', rank)
        energy = comm.allgather(energy)
        forces = comm.allgather(forces)
        # sigmas = comm.allgather(sigmas)
        # mpi_print(f'Step 10-a3: {time.time()-time_init}', rank)
        E_step_filtered = [jtem for item in energy if len(item) != 0 for jtem in item]
        F_step_filtered = [jtem for item in forces if len(item) != 0 for jtem in item]
        # sigma_filtered = [jtem for item in sigmas if len(item) != 0 for jtem in item]

        F_step_avg = np.average(F_step_filtered, axis=0)
        F_step_norm = np.array([[np.linalg.norm(Fcomp) for Fcomp in Ftems] for Ftems in F_step_filtered - F_step_avg])
        F_step_norm_std = np.sqrt(np.average(F_step_norm ** 2, axis=0))
        F_step_norm_avg = np.linalg.norm(F_step_avg, axis=1)

        uncert_coeff = np.zeros(len(struc.get_atomic_numbers()))
        uncert_max_idx = np.argmax(F_step_norm_std)

        # print(f'{F_step_norm_std[uncert_max_idx]}\t{criteria.Un_Abs_F_avg_i}\t{criteria.Un_Abs_F_std_i * 0.5}')
        # print(F_step_norm_std[uncert_max_idx] - criteria.Un_Abs_F_avg_i + criteria.Un_Abs_F_std_i * 0.5)

        # sigma_step_avg = np.average(sigma_filtered, axis=0)

        from libs.lib_util import get_displacements, get_fc_ha, get_E_ha
        displacements = get_displacements(struc.get_positions(), 'geometry.in.supercell')
        F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
        E_ha = get_E_ha(displacements, F_ha)

        E_step = np.average(E_step_filtered)

        if (F_step_norm_std[uncert_max_idx] - criteria.Un_Abs_F_avg_i + criteria.Un_Abs_F_std_i * 0.5) >= 0:
            uncert_relat = (F_step_norm_std[uncert_max_idx] - criteria.Un_Abs_F_avg_i + criteria.Un_Abs_F_std_i * 0.5) / (criteria.Un_Abs_F_std_i * 1.5)
            mpi_print(np.absolute(E_step - E_ha) / E_ha, rank)
            mpi_print(np.absolute(E_step - E_ha), rank)
            # mpi_print(E_ha, rank)
            mpi_print(E_step, rank)
            # mpi_print(criteria.Epotential_avg * (1+meta_Ediff), rank)
            mpi_print(criteria.Epotential_avg, rank)
            mpi_print(f'{np.absolute(E_step - E_ha) / E_ha > meta_Ediff}', rank)
            # mpi_print(f'{E_ha >= criteria.Epotential_avg * (1+meta_Ediff)}', rank)
            mpi_print(f'{E_step >= criteria.Epotential_avg + 3 * criteria.Epotential_std}', rank)
            # mpi_print(criteria.Epotential_avg - 2 * criteria.Epotential_std, rank)

            if np.absolute(E_step - E_ha) / E_ha > meta_Ediff: # or E_step >= criteria.Epotential_avg + 3 * criteria.Epotential_std:
                mpi_print('deactivate', rank)
                uncert_coeff[uncert_max_idx] = 0
            else:
                mpi_print('activate', rank)
                if uncert_relat > 1.0:
                    uncert_coeff[uncert_max_idx] = 1.0
                else:
                    uncert_coeff[uncert_max_idx] = uncert_relat


        # sigma_step_avg = np.average(sigma_filtered, axis=0)

        # if (F_step_norm_std[uncert_max_idx] - criteria.Un_Abs_F_avg_i + criteria.Un_Abs_F_std_i * 0.5) >= 0:
        #     uncert_relat = (F_step_norm_std[uncert_max_idx] - criteria.Un_Abs_F_avg_i + criteria.Un_Abs_F_std_i * 0.5) / (criteria.Un_Abs_F_std_i * 2)
        #     # print(f'{sigma_step_avg / sigma_avg}:{sigma_step_avg}:{sigma_avg}')
        #     if sigma_step_avg / sigma_avg > 2.0:
        #         print('deactivate')
        #         uncert_coeff[uncert_max_idx] = 0
        #     else:
        #         print('activate')
        #         if uncert_relat > 1.0:
        #             uncert_coeff[uncert_max_idx] = 1
        #         else:
        #             uncert_coeff[uncert_max_idx] = uncert_relat


        # sigma_step_avg = np.average(sigma_filtered, axis=0)

        # if (F_step_norm_std[uncert_max_idx] - criteria.Un_Abs_F_avg_i + criteria.Un_Abs_F_std_i * 0.5) >= 0:
        #     uncert_relat = (F_step_norm_std[uncert_max_idx] - criteria.Un_Abs_F_avg_i + criteria.Un_Abs_F_std_i * 0.5) / (criteria.Un_Abs_F_std_i * 2)
        #     if uncert_relat > 1.0 and uncert_relat <= 3.0:
        #         uncert_coeff[uncert_max_idx] = 1
        #     elif uncert_relat > 3.0:
        #         uncert_coeff[uncert_max_idx] = 0
        #     else:
        #         uncert_coeff[uncert_max_idx] = uncert_relat


        # if (F_step_norm_std[uncert_max_idx] - criteria.Un_Abs_F_avg_i + criteria.Un_Abs_F_std_i * 0.5) >= 0:
        #     uncert_relat = (F_step_norm_std[uncert_max_idx] - criteria.Un_Abs_F_avg_i + criteria.Un_Abs_F_std_i * 0.5) / (criteria.Un_Abs_F_std_i * 2)
        #     if uncert_relat > 1.0 and uncert_relat <= 3.0:
        #         uncert_coeff[uncert_max_idx] = 1
        #     else:
        #         uncert_coeff[uncert_max_idx] = uncert_relat


        # sigma_step_avg = np.average(sigma_filtered, axis=0)
        # real_sigma = 0.4

        # print(item - criteria.Un_Abs_F_avg_i + criteria.Un_Abs_F_std_i * 0.5)
        # uncert_coeff = []
        # for item in F_step_norm_std:
        #     if (item - criteria.Un_Abs_F_avg_i + criteria.Un_Abs_F_std_i * 0.5) >= 0:
        #         uncert_relat = (item - criteria.Un_Abs_F_avg_i + criteria.Un_Abs_F_std_i * 0.5) / (criteria.Un_Abs_F_std_i * 2.0) * sigma_step_avg / real_sigma
        #         if uncert_relat > 1.0 and uncert_relat <= 2.0:
        #             uncert_coeff.append(1.0)
        #         elif uncert_relat > 2.0:
        #             uncert_coeff.append(0)
        #         else:
        #             uncert_coeff.append(uncert_relat)
        #     else:
        #         uncert_coeff.append(0)

        # mpi_print(f'Step 10-b: {time.time()-time_init}', rank)

        # from libs.lib_util import get_displacements, get_fc_ha
        # displacements = get_displacements(struc.get_positions(), 'geometry.in.supercell')
        # F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')

        # print(f'uncert_coeff:{uncert_coeff}')
        # comm.Barrier()

        F_ha_rev = []
        for item, jtem in zip(uncert_coeff, F_ha):
            F_ha_rev.append(item * np.array(jtem))

        # mpi_print(f'Step 10-c: {time.time()-time_init}', rank)
        force_avg = F_step_avg - np.array(F_ha_rev)

        # print(uncert_coeff)
        # print(force_avg)

    else:
        forces = None
        if rank == 0:
            struc.calc = calculator
            forces = struc.get_forces(md=True)
        force_avg = comm.bcast(forces, root=0)

    # mpi_print(f'Step 10-d: {time.time()-time_init}', rank)

    return force_avg
