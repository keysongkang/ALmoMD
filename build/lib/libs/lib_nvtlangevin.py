from ase.io.trajectory import TrajectoryWriter
import ase.units as units

import numpy as np
from decimal import Decimal


def NVTLangevin(
    struc, timestep, temperature, friction, steps, loginterval, logfile,
    trajectory, nstep, nmodel, calculator, comm, size, rank, fix_com=True,
):

    if rank == 0:
        file_log = open(logfile, 'w')
        file_log.write(
            'Time[ps]   \tEtot[eV]   \tEpot[eV]    \tEkin[eV]   \tTemperature[K]\n'
        )
        file_log.close()
        file_traj = TrajectoryWriter(filename=trajectory, mode='w')
    
    info_TE, info_PE, info_KE, info_T\
    = get_MDinfo_temp(struc, nstep, nmodel, calculator, comm, size, rank)

    if rank == 0:
        file_log = open(logfile, 'a')
        file_log.write('{:.5f}'.format(Decimal(str(0.0))) + str('   \t') +\
                       '{:.5e}'.format(Decimal(str(info_TE))) + str('\t') +\
                       '{:.5e}'.format(Decimal(str(info_PE))) + str('\t') +\
                       '{:.5e}'.format(Decimal(str(info_KE))) + str('\t') +\
                       '{:.2f}'.format(Decimal(str(info_T))) + str('\n'))
        file_log.close()
        file_traj.write(atoms=struc)

    forces = get_forces(struc, nstep, nmodel, calculator, comm, size, rank)
    
    for index in range(steps):
        natoms = len(struc)
        masses = get_masses(struc.get_masses(), natoms)
        sigma = np.sqrt(2 * temperature * friction / masses)

        c1 = timestep / 2. - timestep * timestep * friction / 8.
        c2 = timestep * friction / 2 - timestep * timestep * friction * friction / 8.
        c3 = np.sqrt(timestep) * sigma / 2. - timestep**1.5 * friction * sigma / 8.
        c5 = timestep**1.5 * sigma / (2 * np.sqrt(3))
        c4 = friction / 2. * c5
        
        if forces is None:
            forces = get_forces(struc, nstep, nmodel, calculator, comm, size, rank)
        velocity = struc.get_velocities()
        
        xi = np.empty(shape=(natoms, 3))
        eta = np.empty(shape=(natoms, 3))
        if rank == 0:
            xi = np.random.standard_normal(size=(natoms, 3))
            eta = np.random.standard_normal(size=(natoms, 3))
        comm.Bcast(xi, root=0)
        comm.Bcast(eta, root=0)
        
        rnd_pos = c5 * eta
        rnd_vel = c3 * xi - c4 * eta
        
        if fix_com:
            rnd_pos -= rnd_pos.sum(axis=0) / natoms
            rnd_vel -= (rnd_vel * masses).sum(axis=0) / (masses * natoms)
            
        # First halfstep in the velocity.
        velocity += (c1 * forces / masses - c2 * velocity + rnd_vel)
        
        # Full step in positions
        position = struc.get_positions()
        
        # Step: x^n -> x^(n+1) - this applies constraints if any.
        struc.set_positions(position + timestep * velocity + rnd_pos)

        # recalc velocities after RATTLE constraints are applied
        velocity = (struc.get_positions() - position - rnd_pos) / timestep
        forces = get_forces(struc, nstep, nmodel, calculator, comm, size, rank)
        
        # Update the velocities
        velocity += (c1 * forces / masses - c2 * velocity + rnd_vel)

        # Second part of RATTLE taken care of here
        struc.set_momenta(velocity * masses)
        
        if index % loginterval == 0:
            info_TE, info_PE, info_KE, info_T\
            = get_MDinfo_temp(struc, nstep, nmodel, calculator, comm, size, rank)
            if rank == 0:
                file_log = open(logfile, 'a')
                simtime = timestep*(index+loginterval)/units.fs/1000
                file_log.write(
                    '{:.5f}'.format(Decimal(str(simtime))) + str('   \t') +\
                    '{:.5e}'.format(Decimal(str(info_TE))) + str('\t') +\
                    '{:.5e}'.format(Decimal(str(info_PE))) + str('\t') +\
                    '{:.5e}'.format(Decimal(str(info_KE))) + str('\t') +\
                    '{:.2f}'.format(Decimal(str(info_T)))  + str('\n')
                )
                file_log.close()
                file_traj.write(atoms=struc)

                
def get_forces(
    struc, nstep, nmodel, calculator, comm, size, rank
):
    if type(calculator) == list:
        forces = []
        zndex = 0
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                if (index_nmodel*nstep + index_nstep) % size == rank:
                    struc.calc = calculator[zndex]
                    forces.append(struc.get_forces(md=True))
                    zndex += 1
        forces = comm.allgather(forces)
        force_avg =\
        np.average([jtem for item in forces if len(item) != 0 for jtem in item],axis=0)
    else:
        forces = None
        if rank == 0:
            struc.calc = calculator
            forces = struc.get_forces(md=True)
        force_avg = comm.bcast(forces, root=0)

    return force_avg


def get_MDinfo_temp(
    struc, nstep, nmodel, calculator, comm, size, rank
):
    if type(calculator) == list: 
        info_TE, info_PE, info_KE, info_T = [], [], [], []
        zndex = 0
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                if (index_nmodel*nstep + index_nstep) % size == rank:
                    struc.calc = calculator[zndex]
                    info_TE.append(struc.get_total_energy())
                    info_PE.append(struc.get_potential_energy())
                    info_KE.append(struc.get_kinetic_energy())
                    info_T.append(struc.get_temperature())
                    zndex += 1
        info_TE = comm.allgather(info_TE)
        info_PE = comm.allgather(info_PE)
        info_KE = comm.allgather(info_KE)
        info_T = comm.allgather(info_T)
        
        info_TE_avg =\
        np.average(np.array([i for items in info_TE for i in items]), axis=0)
        info_PE_avg =\
        np.average(np.array([i for items in info_PE for i in items]), axis=0)
        info_KE_avg =\
        np.average(np.array([i for items in info_KE for i in items]), axis=0)
        info_T_avg =\
        np.average(np.array([i for items in info_T for i in items]), axis=0)
    else:
        info_TE, info_PE, info_KE, info_T = None, None, None, None
        if rank == 0:
            struc.calc = calculator
            info_TE = struc.get_total_energy()
            info_PE = struc.get_potential_energy()
            info_KE = struc.get_kinetic_energy()
            info_T = struc.get_temperature()
        info_TE_avg = comm.bcast(info_TE, root=0)
        info_PE_avg = comm.bcast(info_PE, root=0)
        info_KE_avg = comm.bcast(info_KE, root=0)
        info_T_avg = comm.bcast(info_T, root=0)
                      
    return info_TE_avg, info_PE_avg, info_KE_avg, info_T_avg


def get_masses(get_masses, natoms):
    masses = []
    for index in range(natoms):
        masses.append([get_masses[index]])
    return np.array(masses)