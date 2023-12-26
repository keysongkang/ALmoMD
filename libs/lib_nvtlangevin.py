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
from libs.lib_MD_util import get_forces, get_MDinfo_temp, get_masses
from libs.lib_criteria import eval_uncert, uncert_strconvter

import torch
torch.set_default_dtype(torch.float64)

def NVTLangevin(
    struc, timestep, temperature, friction, steps, loginterval,
    nstep, nmodel, calculator, E_ref, al_type, trajectory, harmonic_F=False,
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
        Langevin_idx = (len(traj_old)-1) * loginterval
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

    # mpi_print(f'Step 2: {time.time()-time_init}', rank)

    # Get averaged force from trained models
    try:
        forces = struc.get_forces()
    except Exception as e:
        forces = get_forces(struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F)

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
            forces = get_forces(struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F)
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
        forces = get_forces(struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F)
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
