import numpy as np

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

    # mpi_print(f'Step 10-a: {time.time()-time_init}', rank)
    if type(calculator) == list:
        forces = []
        zndex = 0
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                # mpi_print(f'Step 10-a1 first {rank}: {time.time()-time_init}', rank)
                struc.calc = calculator[zndex]
                # mpi_print(f'Step 10-a1 second {rank}: {time.time()-time_init}', rank)
                temp_force = struc.get_forces()
                # mpi_print(f'Step 10-a1 third {rank}: {time.time()-time_init}', rank)
                forces.append(temp_force)
                # mpi_print(f'Step 10-a1 last {rank}: {time.time()-time_init}', rank)
                zndex += 1

        # mpi_print(f'Step 10-b: {time.time()-time_init}', rank)
        if harmonic_F and anharmonic_F:
            from libs.lib_util import get_displacements, get_fc_ha
            displacements = get_displacements(struc.get_positions(), 'geometry.in.supercell')
            F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
            forces = forces + F_ha

        # mpi_print(f'Step 10-c: {time.time()-time_init}', rank)
        force_avg = np.average(forces, axis=0)

    else:
        struc.calc = calculator
        forces_avg = struc.get_forces(md=True)

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

    if type(calculator) == list:
        stress = []
        zndex = 0
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                # mpi_print(f'Step 10-a1 first {rank}: {time.time()-time_init}', rank)
                struc.calc = calculator[zndex]
                # mpi_print(f'Step 10-a1 second {rank}: {time.time()-time_init}', rank)
                temp_stress = struc.get_stress(include_ideal_gas=True)
                # mpi_print(f'Step 10-a1 third {rank}: {time.time()-time_init}', rank)
                stress.append(temp_stress)
                # mpi_print(f'Step 10-a1 last {rank}: {time.time()-time_init}', rank)
                zndex += 1

        # mpi_print(f'Step 10-c: {time.time()-time_init}', rank)
        stress_avg = np.average(stress, axis=0)

    else:
        struc.calc = calculator
        stress_avg = struc.get_stress(include_ideal_gas=True)

    return stress_avg


def get_MDinfo_temp(
    struc, nstep, nmodel, calculator, harmonic_F, E_ref, signal_P = False
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

    # Get calculators from trained models and corresponding predicted quantities
    info_TE, info_PE, info_KE, info_T = [], [], [], []
    if signal_P:
        info_P = []

    zndex = 0
    for index_nmodel in range(nmodel):
        for index_nstep in range(nstep):
            struc.calc = calculator[zndex]
            PE = struc.get_potential_energy() - E_ref[0][zndex]
            KE = struc.get_kinetic_energy()
            TE = PE + KE
            info_TE.append(TE)
            info_PE.append(PE)
            info_KE.append(KE)
            info_T.append(struc.get_temperature())
            if signal_P:
                info_P.append(struc.get_stress())
            zndex += 1
    
    # Get their average
    # info_TE_avg =\
    # np.average(np.array([i for items in info_TE for i in items]), axis=0)
    info_TE_avg = np.average(info_TE, axis=0)

    if harmonic_F:
        from libs.lib_util import get_displacements, get_fc_ha, get_E_ha
        displacements = get_displacements(struc.get_positions(), 'geometry.in.supercell')
        F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
        E_ha = get_E_ha(displacements, fc_ha)
        info_PE_avg = np.average(info_PE, axis=0) + E_ha
    else:
        info_PE_avg = np.average(info_PE, axis=0)

    info_KE_avg = np.average(info_KE, axis=0)
    info_T_avg = np.average(info_T, axis=0)

    if signal_P:
        from ase import units
        info_P_avg = np.average(info_P, axis=0)
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