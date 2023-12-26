import numpy as np
from mpi4py import MPI

import torch
torch.set_default_dtype(torch.float64)

def get_forces(
    struc, nstep, nmodel, calculator, harmonic_F, anharmonic_F
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

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # mpi_print(f'Step 10-a: {time.time()-time_init}', rank)
    if type(calculator) == list:
        forces = []
        zndex = 0
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                if (index_nmodel*nstep + index_nstep) % size == rank:
                    # mpi_print(f'Step 10-a1 first {rank}: {time.time()-time_init}', rank)
                    struc.calc = calculator[zndex]
                    # mpi_print(f'Step 10-a1 second {rank}: {time.time()-time_init}', rank)
                    temp_force = struc.get_forces()
                    # mpi_print(f'Step 10-a1 third {rank}: {time.time()-time_init}', rank)
                    forces.append(temp_force)
                    # mpi_print(f'Step 10-a1 last {rank}: {time.time()-time_init}', rank)
                    zndex += 1
        # mpi_print(f'Step 10-a2: {time.time()-time_init}', rank)
        forces = comm.allgather(forces)
        # mpi_print(f'Step 10-a3: {time.time()-time_init}', rank)
        F_step_filtered = [jtem for item in forces if len(item) != 0 for jtem in item]

        # mpi_print(f'Step 10-b: {time.time()-time_init}', rank)
        if harmonic_F and anharmonic_F:
            from libs.lib_util import get_displacements, get_fc_ha
            displacements = get_displacements(struc.get_positions(), 'geometry.in.supercell')
            F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
            F_step_filtered = F_step_filtered + F_ha

        # mpi_print(f'Step 10-c: {time.time()-time_init}', rank)
        force_avg = np.average(F_step_filtered, axis=0)

    else:
        forces = None
        if rank == 0:
            struc.calc = calculator
            forces = struc.get_forces(md=True)
        force_avg = comm.bcast(forces, root=0)

    # mpi_print(f'Step 10-d: {time.time()-time_init}', rank)

    return force_avg


def get_stress(
    struc, nstep, nmodel, calculator
):
    """Function [get_stress]
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

    if type(calculator) == list:
        stress = []
        zndex = 0
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                if (index_nmodel*nstep + index_nstep) % size == rank:
                    # mpi_print(f'Step 10-a1 first {rank}: {time.time()-time_init}', rank)
                    struc.calc = calculator[zndex]
                    # mpi_print(f'Step 10-a1 second {rank}: {time.time()-time_init}', rank)
                    temp_stress = struc.get_stress(include_ideal_gas=True)
                    # mpi_print(f'Step 10-a1 third {rank}: {time.time()-time_init}', rank)
                    stress.append(temp_stress)
                    # mpi_print(f'Step 10-a1 last {rank}: {time.time()-time_init}', rank)
                    zndex += 1
        # mpi_print(f'Step 10-a2: {time.time()-time_init}', rank)
        stress = comm.allgather(stress)
        # mpi_print(f'Step 10-a3: {time.time()-time_init}', rank)
        stress_step_filtered = [jtem for item in stress if len(item) != 0 for jtem in item]

        # mpi_print(f'Step 10-c: {time.time()-time_init}', rank)
        stress_avg = np.average(stress_step_filtered, axis=0)

    else:
        stress = None
        if rank == 0:
            struc.calc = calculator
            stress = struc.get_stress(include_ideal_gas=True)
        stress_avg = comm.bcast(stress, root=0)

    return stress_avg


def get_MDinfo_temp(
    struc, nstep, nmodel, calculator, harmonic_F, signal_P = False
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

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Get calculators from trained models and corresponding predicted quantities
    info_TE, info_PE, info_KE, info_T = [], [], [], []
    if signal_P:
        info_P = []

    zndex = 0
    for index_nmodel in range(nmodel):
        for index_nstep in range(nstep):
            if (index_nmodel*nstep + index_nstep) % size == rank:
                struc.calc = calculator[zndex]
                PE = struc.get_potential_energy()
                KE = struc.get_kinetic_energy()
                TE = PE + KE
                info_TE.append(TE)
                info_PE.append(PE)
                info_KE.append(KE)
                info_T.append(struc.get_temperature())
                if signal_P:
                    info_P.append(struc.get_stress())
                zndex += 1

    info_TE = comm.allgather(info_TE)
    info_PE = comm.allgather(info_PE)
    info_KE = comm.allgather(info_KE)
    info_T = comm.allgather(info_T)
    if signal_P:
        info_P = comm.allgather(info_P)
    
    # Get their average
    info_TE_avg =\
    np.average(np.array([i for items in info_TE for i in items]), axis=0)

    if harmonic_F:
        from libs.lib_util import get_displacements, get_fc_ha, get_E_ha
        displacements = get_displacements(struc.get_positions(), 'geometry.in.supercell')
        F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
        E_ha = get_E_ha(displacements, fc_ha)
        info_PE_avg =\
        np.average(np.array([i for items in info_PE for i in items]), axis=0) + E_ha
    else:
        info_PE_avg =\
        np.average(np.array([i for items in info_PE for i in items]), axis=0)

    info_KE_avg =\
    np.average(np.array([i for items in info_KE for i in items]), axis=0)
    info_T_avg =\
    np.average(np.array([i for items in info_T for i in items]), axis=0)

    if signal_P:
        from ase import units
        info_P_avg =\
        np.average(np.array([i for items in info_P for i in items]), axis=0)
        return info_TE_avg, info_PE_avg, info_KE_avg, info_T_avg, np.average(info_P_avg[:3])/units.GPa
                      
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