from ase.io.trajectory import Trajectory
from ase.io.trajectory import TrajectoryWriter
import ase.units as units
from ase.io.cif        import write_cif

import time
import os
import numpy as np
from mpi4py import MPI
from decimal import Decimal

from libs.lib_util    import single_print, mpi_print
from libs.lib_MD_util import get_MDinfo_temp, get_masses
from libs.lib_criteria import eval_uncert, uncert_strconvter, get_criteria

import torch
torch.set_default_dtype(torch.float64)

def NVTLangevin_meta(
    struc, timestep, temperature, friction, steps, loginterval,
    nstep, nmodel, calculator, E_ref, al_type, trajectory, meta_Ediff, harmonic_F=False,
    anharmonic_F=False, logfile=None, signal_uncert=False, signal_append=True, fix_com=True,
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
    Langevin_idx = 0

    # mpi_print(f'Step 1: {time.time()-time_init}', rank)
    if signal_append and os.path.exists(trajectory) and os.path.getsize(trajectory) != 0: # If appending and the file exists,
        # Read the previous trajectory
        traj_old = Trajectory(
            trajectory,
            properties=['forces', 'velocities', 'temperature']
            )
        # Get the index
        Langevin_idx = len(traj_old) * loginterval
        # Get the last structure
        struc = traj_old[-1]
        file_traj = TrajectoryWriter(filename=trajectory, mode='a')
    else: # New start
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
                struc, nstep, nmodel, calculator, harmonic_F
                )

            if signal_uncert:
                # Get absolute and relative uncertainties of energy and force
                # and also total energy
                uncerts, Epot_step, S_step =\
                eval_uncert(struc, nstep, nmodel, E_ref, calculator, al_type, harmonic_F)

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

    # Put random 0 and 300
    criteria = get_criteria(temperature, 0, 0, 300, al_type)

    # mpi_print(f'Step 2: {time.time()-time_init}', rank)

    # Get averaged force from trained models
    # try:
    #     forces = struc.get_forces()
    # except Exception as e:
    #     forces = get_forces_meta(struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F, criteria, meta_Ediff)

    forces = get_forces_meta(struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F, criteria, meta_Ediff)

    # mpi_print(f'Step 3: {time.time()-time_init}', rank)
    # Go trough steps until the requested number of steps
    # If appending, it starts from Langevin_idx. Otherwise, Langevin_idx = 0
    for idx in range(Langevin_idx, steps):

        # mpi_print(f'Step 4: {time.time()-time_init}', rank)
        # Get essential properties
        natoms = len(struc)
        masses = get_masses(struc.get_masses(), natoms)
        sigma = np.sqrt(2 * temperature * friction / masses)

        # mpi_print(f'Step 5: {time.time()-time_init}', rank)
        # Get Langevin coefficients
        c1 = timestep / 2. - timestep * timestep * friction / 8.
        c2 = timestep * friction / 2 - timestep * timestep * friction * friction / 8.
        c3 = np.sqrt(timestep) * sigma / 2. - timestep**1.5 * friction * sigma / 8.
        c5 = timestep**1.5 * sigma / (2 * np.sqrt(3))
        c4 = friction / 2. * c5

        # mpi_print(f'Step 6: {time.time()-time_init}', rank)
        # Get averaged forces and velocities
        if forces is None:
            forces = get_forces_meta(struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F, criteria, meta_Ediff)
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
        forces = get_forces_meta(struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F, criteria, meta_Ediff)
        comm.Barrier()
        
        # mpi_print(f'Step 10-2: {time.time()-time_init}', rank)
        # Update the velocities
        velocity += (c1 * forces / masses - c2 * velocity + rnd_vel)

        # mpi_print(f'Step 10-3: {time.time()-time_init}', rank)
        # Second part of RATTLE taken care of here
        struc.set_momenta(velocity * masses)
        
        # mpi_print(f'Step 11: {time.time()-time_init}', rank)
        # Log MD information at regular intervals
        if idx % loginterval == 0:
            if isinstance(logfile, str):
                # mpi_print(f'Step 12: {time.time()-time_init}', rank)
                info_TE, info_PE, info_KE, info_T = get_MDinfo_temp(
                    struc, nstep, nmodel, calculator, harmonic_F
                    )

                # mpi_print(f'Step 13: {time.time()-time_init}', rank)
                if signal_uncert:
                    # Get absolute and relative uncertainties of energy and force
                    # and also total energy
                    uncerts, Epot_step, S_step =\
                    eval_uncert(struc, nstep, nmodel, E_ref, calculator, al_type, harmonic_F)

                # mpi_print(f'Step 14: {time.time()-time_init}', rank)
                if rank == 0:
                    file_log = open(logfile, 'a')
                    simtime = timestep*(idx+loginterval)/units.fs/1000
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


# def get_MDinfo_temp(
#     struc, nstep, nmodel, calculator, harmonic_F
# ):
#     """Function [get_MDinfo_temp]
#     Extract the average of total, potential, and kinetic energy of
#     a structral configuration from the Molecular Dynamics

#     Parameters:

#     struc_step: ASE atoms
#         A structral configuration at the current step
#     nstep: int
#         The number of subsampling sets
#     nmodel: int
#         The number of ensemble model sets with different initialization
#     calculator: ASE calculator or list of ASE calculators
#         Calculators from trained models

#     Returns:

#     info_TE_avg: float
#         Averaged total energy across trained models
#     info_PE_avg: float
#         Averaged potential energy across trained models
#     info_KE_avg: float
#         Averaged kinetic energy across trained models
#     info_T_avg: float
#         Averaged temeprature across trained models
#     """

#     # Extract MPI infos
#     comm = MPI.COMM_WORLD
#     size = comm.Get_size()
#     rank = comm.Get_rank()

#     # Get calculators from trained models and corresponding predicted quantities
#     info_TE, info_PE, info_KE, info_T = [], [], [], []
#     zndex = 0
#     for index_nmodel in range(nmodel):
#         for index_nstep in range(nstep):
#             if (index_nmodel*nstep + index_nstep) % size == rank:
#                 struc.calc = calculator[zndex]
#                 PE = struc.get_potential_energy()
#                 KE = struc.get_kinetic_energy()
#                 TE = PE + KE
#                 info_TE.append(TE)
#                 info_PE.append(PE)
#                 info_KE.append(KE)
#                 info_T.append(struc.get_temperature())
#                 zndex += 1
#     info_TE = comm.allgather(info_TE)
#     info_PE = comm.allgather(info_PE)
#     info_KE = comm.allgather(info_KE)
#     info_T = comm.allgather(info_T)
    
#     # Get their average
#     info_TE_avg =\
#     np.average(np.array([i for items in info_TE for i in items]), axis=0)

#     if harmonic_F:
#         from libs.lib_util import get_displacements, get_fc_ha, get_E_ha
#         displacements = get_displacements(struc.get_positions(), 'geometry.in.supercell')
#         F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
#         E_ha = get_E_ha(displacements, fc_ha)
#         info_PE_avg =\
#         np.average(np.array([i for items in info_PE for i in items]), axis=0) + E_ha
#     else:
#         info_PE_avg =\
#         np.average(np.array([i for items in info_PE for i in items]), axis=0)

#     info_KE_avg =\
#     np.average(np.array([i for items in info_KE for i in items]), axis=0)
#     info_T_avg =\
#     np.average(np.array([i for items in info_T for i in items]), axis=0)
                      
#     return info_TE_avg, info_PE_avg, info_KE_avg, info_T_avg


# def get_masses(get_masses, natoms):
#     """Function [get_masses]
#     Extract the list of atoms' mass

#     Parameters:

#     get_masses: np.array of float
#         An array of masses of elements
#     natoms: int
#         The number of atoms in the simulation cell

#     Returns:

#     masses: float
#         An array of atoms' masses in the simulation cell
#     """

#     masses = []
#     for idx in range(natoms):
#         masses.append([get_masses[idx]])

#     return np.array(masses)