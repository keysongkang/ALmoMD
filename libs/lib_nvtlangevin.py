from ase.io.trajectory import TrajectoryWriter
import ase.units as units

import numpy as np
from mpi4py import MPI
from decimal import Decimal


def NVTLangevin(
    struc, timestep, temperature, friction, steps, loginterval, logfile,
    trajectory, nstep, nmodel, calculator, signal_append=True, fix_com=True,
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

    logfile: str
        A name of MD logfile
    trajectory: str
        A name of MD trajectory file
    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization
    calculator: ASE calculator
        Any calculator
    fixcm: bool (optional)
        If True, the position and momentum of the center of mass is
        kept unperturbed.  Default: True.
    """

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Initialization of index
    Langevin_idx = 0

    if signal_append and os.path.exist(trajectory): # If appending and the file exists,
        # Read the previous trajectory
        traj_old = np.load(trajectory)
        # Get the index
        Langevin_idx = len(traj_old) * loginterval
        # Get the last structure
        struc = traj_old[-1]
    else: # New start
        # Prepare a log file ##!! Need to be optional
        if rank == 0:
            file_log = open(logfile, 'w')
            file_log.write(
                'Time[ps]   \tEtot[eV]   \tEpot[eV]    \tEkin[eV]   \tTemperature[K]\n'
            )
            file_log.close()
            file_traj = TrajectoryWriter(filename=trajectory, mode='w')
        
        # Get MD information at the current step
        info_TE, info_PE, info_KE, info_T = get_MDinfo_temp(
            struc, nstep, nmodel, calculator
            )

        # Log MD information at the current step in the log file
        if rank == 0:
            file_log = open(logfile, 'a')
            file_log.write('{:.5f}'.format(Decimal(str(0.0))) + str('   \t') +\
                           '{:.5e}'.format(Decimal(str(info_TE))) + str('\t') +\
                           '{:.5e}'.format(Decimal(str(info_PE))) + str('\t') +\
                           '{:.5e}'.format(Decimal(str(info_KE))) + str('\t') +\
                           '{:.2f}'.format(Decimal(str(info_T))) + str('\n'))
            file_log.close()
            file_traj.write(atoms=struc)

    # Get averaged force from trained models
    forces = get_forces(struc, nstep, nmodel, calculator)

    # Go trough steps until the requested number of steps
    # If appending, it starts from Langevin_idx. Otherwise, Langevin_idx = 0
    for idx in range(Langevin_idx, steps):
        # Get essential properties
        natoms = len(struc)
        masses = get_masses(struc.get_masses(), natoms)
        sigma = np.sqrt(2 * temperature * friction / masses)

        # Get Langevin coefficients
        c1 = timestep / 2. - timestep * timestep * friction / 8.
        c2 = timestep * friction / 2 - timestep * timestep * friction * friction / 8.
        c3 = np.sqrt(timestep) * sigma / 2. - timestep**1.5 * friction * sigma / 8.
        c5 = timestep**1.5 * sigma / (2 * np.sqrt(3))
        c4 = friction / 2. * c5
        
        # Get averaged forces and velocities
        if forces is None:
            forces = get_forces(struc, nstep, nmodel, calculator)
        # Velocity is already calculated based on averaged forces
        # in the previous step
        velocity = struc.get_velocities()
        
        # Sample the random numbers for the temperature fluctuation
        xi = np.empty(shape=(natoms, 3))
        eta = np.empty(shape=(natoms, 3))
        if rank == 0:
            xi = np.random.standard_normal(size=(natoms, 3))
            eta = np.random.standard_normal(size=(natoms, 3))
        comm.Bcast(xi, root=0)
        comm.Bcast(eta, root=0)
        
        # Get get changes of positions and velocities
        rnd_pos = c5 * eta
        rnd_vel = c3 * xi - c4 * eta
        
        # Check the center of mass
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
        forces = get_forces(struc, nstep, nmodel, calculator)
        
        # Update the velocities
        velocity += (c1 * forces / masses - c2 * velocity + rnd_vel)

        # Second part of RATTLE taken care of here
        struc.set_momenta(velocity * masses)
        
        # Log MD information at regular intervals
        if idx % loginterval == 0:
            info_TE, info_PE, info_KE, info_T = get_MDinfo_temp(
                struc, nstep, nmodel, calculator
                )
            if rank == 0:
                file_log = open(logfile, 'a')
                simtime = timestep*(idx+loginterval)/units.fs/1000
                file_log.write(
                    '{:.5f}'.format(Decimal(str(simtime))) + '   \t' +
                    '{:.5e}'.format(Decimal(str(info_TE))) + '\t' +
                    '{:.5e}'.format(Decimal(str(info_PE))) + '\t' +
                    '{:.5e}'.format(Decimal(str(info_KE))) + '\t' +
                    '{:.2f}'.format(Decimal(str(info_T)))  + '\n'
                )
                file_log.close()
                file_traj.write(atoms=struc)

                
def get_forces(
    struc, nstep, nmodel, calculator
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

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Get calculators from trained models and corresponding predicted forces
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
        # Get the average
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
    struc, nstep, nmodel, calculator
):
    """Function [get_MDinfo_temp]
    Extract the average of total, potential, and kinetic energy of
    a structral configuration from the Molecular Dynamics

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

    info_TE_avg: float
        Averaged total energy across trained models
    info_PE_avg: float
        Averaged potential energy across trained models
    info_KE_avg: float
        Averaged kinetic energy across trained models
    info_T_avg: float
        Averaged temeprature across trained models
    """
    ##!! Need to check wich quantity (TE, PE, KE, T) we need.

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Get calculators from trained models and corresponding predicted quantities
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
        
        # Get their average
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
    """Function [get_masses]
    Extract the list of atoms' mass

    Parameters:

    get_masses: np.array of float
        An array of masses of elements
    natoms: int
        The number of atoms in the simulation cell

    Returns:

    masses: float
        An array of atoms' masses in the simulation cell
    """

    masses = []
    for idx in range(natoms):
        masses.append([get_masses[idx]])

    return np.array(masses)