import os
import sys
import numpy as np
import pandas as pd
from mpi4py import MPI
from decimal import Decimal
from scipy import special
from libs.lib_util import single_print


def eval_uncert(
    struc_step, nstep, nmodel, E_ref, calculator, al_type
):
    """Function [eval_uncert]
    Evalulate the absolute and relative uncertainties of
    predicted energies and forces.

    Parameters:

    struc_step: ASE atoms
        A structral configuration at the current step
    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization
    E_ref: flaot
        The energy of reference state (Here, ground state)
    calculator: ASE calculator
        Calculators from trained models
    al_type: str
        Type of active learning: 'energy', 'force', 'force_max'

    Returns:

    UncertAbs_E: float or str
        Absolute uncertainty of predicted energy
    UncertRel_E: float or str
        Relative uncertainty of predicted energy
    UncertAbs_F: float or str
        Absolute uncertainty of predicted force
    UncertRel_F: float or str
        Relative uncertainty of predicted force
    """

    ## Depending on an active learning type (al_type), the format of output changes
    # Active learning based on the uncertainty of predicted energy
    if al_type == 'energy':
        Epot_step_avg, Epot_step_std = eval_uncert_E(
            struc_step, nstep, nmodel, E_ref, calculator, al_type
        )
        
        return (
            Epot_step_std,
            Epot_step_std / Epot_step_avg,
            '----          ',
            '----          ',
            Epot_step_avg
        )

    # Active learning based on the AVERAGED uncertainty of predicted force
    elif al_type == 'force':
        # Just get potential energy for the ensemble probability
        Epot_step_avg, Epot_step_std = eval_uncert_E(
            struc_step, nstep, nmodel, E_ref, calculator, al_type
        )
        F_step_norm_avg, F_step_norm_std = eval_uncert_F(
            struc_step, nstep, nmodel, E_ref, calculator
        )

        return (
            '----          ',
            '----          ',
            np.average(F_step_norm_std),
            np.average(F_step_norm_std / F_step_norm_avg),
            Epot_step_avg
        )

    # Active learning based on the MAXIUM uncertainty of predicted force
    elif al_type == 'force_max':
        # Just get potential energy for the ensemble probability
        Epot_step_avg, Epot_step_std = eval_uncert_E(
            struc_step, nstep, nmodel, E_ref, calculator, al_type
        )
        F_step_norm_avg, F_step_norm_std = eval_uncert_F(
            struc_step, nstep, nmodel, E_ref, calculator
        )

        return (
            '----          ',
            '----          ',
            np.ndarray.max(F_step_norm_std),
            np.ndarray.max(
                np.array([std / avg for avg, std in zip(F_step_norm_avg, F_step_norm_std) if avg > 0.05])
            ),
            Epot_step_avg
        )

    ##!! this part is needed to be check. it might need F instead of Fmax
    elif al_type == 'EandFmax' or al_type == 'EorFmax':
        Epot_step_avg, Epot_step_std, Epot_step_avg = eval_uncert_E(
            struc_step, nstep, nmodel, E_ref, calculator, al_type
        )
        F_step_norm_avg, F_step_norm_std, Epot_step_avg = eval_uncert_F(
            struc_step, nstep, nmodel, E_ref, calculator, al_type
        )

        return (
            Epot_step_std,
            Epot_step_std / Epot_step_avg,
            np.ndarray.max(F_step_norm_std),
            np.ndarray.max(
                np.array([std / avg for avg, std in zip(F_step_norm_avg, F_step_norm_std) if avg > 0.05])
            ),
            Epot_step_avg
        )
    
    else:
        sys.exit("You need to set al_type.")
        



def eval_uncert_E(
    struc_step, nstep, nmodel, E_ref, calculator, al_type
):
    """Function [eval_uncert_E]
    Evalulate the average and standard deviation of predicted energies.

    Parameters:

    struc_step: ASE atoms
        A structral configuration at the current step
    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization
    E_ref: flaot
        The energy of reference state (Here, ground state)
    calculator: ASE calculator
        Calculators from trained models
    al_type: str
        Type of active learning: 'energy', 'force', 'force_max'

    Returns:

    Epot_step_avg: float
        Average of predicted energies
    Epot_step_std: float
        Standard deviation of predicted energies
    """

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Prepare empty lists for potential and total energies
    Epot_step = []
    zndex = 0

    # Get predicted potential and total energies shifted by E_ref (ground state energy)
    for index_nmodel in range(nmodel):
        for index_nstep in range(nstep):
            if (index_nmodel*nstep + index_nstep) % size == rank:
                struc_step.calc = calculator[zndex]
                Epot_step.append(struc_step.get_potential_energy() - E_ref)
                zndex += 1
    Epot_step = comm.allgather(Epot_step)

    # Get the average and standard deviation of predicted potential energies
    # and the average of total energies
    Epot_step_avg =\
    np.average(np.array([i for items in Epot_step for i in items]), axis=0)
    if al_type == 'force' or al_type == 'force_max':
        Epot_step_std = '--------'
    else:
        Epot_step_std =\
        np.std(np.array([i for items in Epot_step for i in items]), axis=0)
    
    return Epot_step_avg, Epot_step_std



def eval_uncert_F(
    struc_step, nstep, nmodel, E_ref, calculator
):
    """Function [eval_uncert_F]
    Evalulate the average and standard deviation of predicted forces.

    Parameters:

    struc_step: ASE atoms
        A structral configuration at the current step
    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization
    E_ref: flaot
        The energy of reference state (Here, ground state)
    calculator: ASE calculator
        Calculators from trained models

    Returns:

    F_step_norm_avg: float
        Average of the norm of predicted forces
    F_step_norm_std: float
        Standard deviation of the norm of predicted forces
    """

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Prepare empty lists for forces and total energies
    F_step = []
    zndex = 0

    # Get predicted forces and total energies shifted by E_ref (ground state energy)
    for index_nmodel in range(nmodel):
        for index_nstep in range(nstep):
            if (index_nmodel*nstep + index_nstep) % size == rank:
                struc_step.calc = calculator[zndex]
                F_step.append(struc_step.get_forces())
                zndex += 1
    F_step = comm.allgather(F_step)

    # Get the average and standard deviation of the norm of predicted forces
    F_step_filtered = np.array([i for items in F_step for i in items])
    F_step_avg = np.average(F_step_filtered, axis=0)
    F_step_norm = np.linalg.norm(F_step_filtered - F_step_avg, axis=1)
    F_step_norm_std = np.sqrt(np.average(F_step_norm ** 2, axis=0))
    F_step_norm_avg = np.linalg.norm(F_step_avg, axis=0)
    
    return F_step_norm_avg, F_step_norm_std


def get_criteria(
    temperature, pressure, index, steps_init
):
    """Function [get_criteria]
    Get average and standard deviation of absolute and relative undertainty
    of energies and forces and also those of total energy
    during the MLMD_init steps

    Parameters:

    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    index: int
        The index of AL interactive step
    steps_init: int
        Initialize MD steps to get averaged uncertainties and energies

    Returns:

    criteria_UncertAbs_E_avg: float
        Average of absolute uncertainty of energies
    criteria_UncertAbs_E_std: float
        Standard deviation of absolute uncertainty of energies
    criteria_UncertRel_E_avg: float
        Average of relative uncertainty of energies
    criteria_UncertRel_E_std: float
        Standard deviation of relative uncertainty of energies
    criteria_UncertAbs_F_avg: float
        Average of absolute uncertainty of forces
    criteria_UncertAbs_F_std: float
        Standard deviation of absolute uncertainty of forces
    criteria_UncertRel_F_avg: float
        Average of relative uncertainty of forces
    criteria_UncertRel_F_std: float
        Standard deviation of relative uncertainty of forces
    criteria_Epot_step_avg: float
        Average of potential energies
    criteria_Epot_step_std: float
        Standard deviation of potential energies
    """

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Read all uncertainty results
    uncert_data = pd.read_csv(
        f'UNCERT/uncertainty-{temperature}K-{pressure}bar_{index}.txt',
        index_col=False, delimiter='\t'
        )
    UncerAbs_E_list = uncert_data.loc[:steps_init, 'UncertAbs_E'].values
    UncerRel_E_list = uncert_data.loc[:steps_init, 'UncertRel_E'].values
    UncerAbs_F_list = uncert_data.loc[:steps_init, 'UncertAbs_F'].values
    UncerRel_F_list = uncert_data.loc[:steps_init, 'UncertRel_F'].values
    Epot_step_list = uncert_data.loc[:steps_init, 'Epot_average'].values
    del uncert_data  # To reduce the memory usage
    
    # Get their average and standard deviation
    criteria_UncertAbs_E_avg = uncert_average(UncerAbs_E_list)
    criteria_UncertAbs_E_std = uncert_std(UncerAbs_E_list)
    criteria_UncertRel_E_avg = uncert_average(UncerRel_E_list)
    criteria_UncertRel_E_std = uncert_std(UncerRel_E_list)
    criteria_UncertAbs_F_avg = uncert_average(UncerAbs_F_list)
    criteria_UncertAbs_F_std = uncert_std(UncerAbs_F_list)
    criteria_UncertRel_F_avg = uncert_average(UncerRel_F_list)
    criteria_UncertRel_F_std = uncert_std(UncerRel_F_list)
    criteria_Epot_step_avg = np.average(Epot_step_list)
    criteria_Epot_step_std = np.std(Epot_step_list)
    
    # Check the contents in 'result.txt' before recording
    if os.path.exists('result.txt'):
        result_data = \
        pd.read_csv('result.txt', index_col=False, delimiter='\t')
        get_criteria_index = result_data.loc[:,'UncerAbs_F_ini'].iloc[-1];
        if get_criteria_index == '----          ':
            get_criteria_index = 0
    else:
        get_criteria_index = np.nan

    # Record when the corresponding slots in 'result.txt' are empty
    if np.isnan(get_criteria_index):
        # Record the average values
        if rank == 0:
            with open('result.txt', 'a') as criteriafile:
                criteriafile.write(
                    uncert_strconvter(criteria_UncertRel_E_avg) + '\t' +
                    uncert_strconvter(criteria_UncertAbs_E_avg) + '\t' +
                    uncert_strconvter(criteria_UncertRel_F_avg) + '\t' +
                    uncert_strconvter(criteria_UncertAbs_F_avg) + '\t'
                )
    
    return (
        criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std,
        criteria_UncertRel_E_avg, criteria_UncertRel_E_std,
        criteria_UncertAbs_F_avg, criteria_UncertAbs_F_std,
        criteria_UncertRel_F_avg, criteria_UncertRel_F_std,
        criteria_Epot_step_avg, criteria_Epot_step_std
    )

    

def get_result(temperature, pressure, index, steps_init):
    """Function [get_result]
    Get average and standard deviation of absolute and relative undertainty
    of energies and forces and also those of total energy for all steps

    Parameters:

    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    index: int
        The index of AL interactive step
    steps_init: int
        Initialize MD steps to get averaged uncertainties and energies
    """

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Read all uncertainty results
    uncert_data = pd.read_csv(
        f'UNCERT/uncertainty-{temperature}K-{pressure}bar_{index}.txt',
        index_col=False, delimiter='\t'
        )
    UncerAbs_E_list = uncert_data.loc[:steps_init,'UncertAbs_E'].values
    UncerRel_E_list = uncert_data.loc[:,'UncertRel_E'].values
    UncerAbs_F_list = uncert_data.loc[:steps_init,'UncertAbs_F'].values
    UncerRel_F_list = uncert_data.loc[:,'UncertRel_F'].values
    del uncert_data  # To reduce the memory usage
    
    # Get their average and standard deviation
    criteria_UncertAbs_E_avg_all = uncert_average(UncerAbs_E_list[:])
    criteria_UncertRel_E_avg_all = uncert_average(UncerRel_E_list[:])
    criteria_UncertAbs_F_avg_all = uncert_average(UncerAbs_F_list[:])
    criteria_UncertRel_F_avg_all = uncert_average(UncerRel_F_list[:])

    # Check the contents in 'result.txt' before recording
    if os.path.exists('result.txt'):
        result_data = \
        pd.read_csv('result.txt', index_col=False, delimiter='\t')
        get_criteria_index = result_data.loc[:,'UncerAbs_F_all'].iloc[-1];
        if get_criteria_index == '----          ':
            get_criteria_index = 0
    else:
        get_criteria_index = np.nan

    # Record when the corresponding slots in 'result.txt' are empty
    if np.isnan(get_criteria_index):
        # Record the average values
        if rank == 0:
            with open('result.txt', 'a') as criteriafile:
                criteriafile.write(
                    uncert_strconvter(criteria_UncertRel_E_avg_all) + '\t' +
                    uncert_strconvter(criteria_UncertAbs_E_avg_all) + '\t' +
                    uncert_strconvter(criteria_UncertRel_F_avg_all) + '\t' +
                    uncert_strconvter(criteria_UncertAbs_F_avg_all) + '\n'
                )

    
def uncert_average(itemlist):
    """Function [uncert_average]
    If input is list of float values, return their average.
    Otherwise, return a string with the dashed line.

    Parameters:

    itemlist: list of float or str
        list of float values or string
    """
    return '----          ' if itemlist[0] == '----          ' else np.average(itemlist)
    
    
def uncert_std(itemlist):
    """Function [uncert_std]
    If input is list of float values, return their standard deviation.
    Otherwise, return a string with the dashed line.

    Parameters:

    itemlist: list of float or str
        list of float values or string
    """
    return '----          ' if itemlist[0] == '----          ' else np.std(itemlist)

def uncert_strconvter(value):
    """Function [uncert_strconvter]
    If the input is a string, it will be returned as is.
    Otherwise, a float number will be returned in scientific format,
    with five significant digits.

    Parameters:

    value: float or str
        any input
    """
    if value == '----          ':
        return value
    return '{:.5e}'.format(Decimal(value))
    
    
def get_criteria_prob(
    al_type, uncert_type, uncert_shift, uncert_grad,
    kB, NumAtoms, temperature, 
    Epot_step, criteria_Epot_step_avg, criteria_Epot_step_std,
    UncertAbs_E, criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std,
    UncertRel_E, criteria_UncertRel_E_avg, criteria_UncertRel_E_std,
    UncertAbs_F, criteria_UncertAbs_F_avg, criteria_UncertAbs_F_std,
    UncertRel_F, criteria_UncertRel_F_avg, criteria_UncertRel_F_std
):
    """Function [get_criteria_prob]
    Utilize the average and standard deviation obtained from 'get_criteria'
    to calculate the probability of satisfying the acceptance criteria.
    Probability has three parts;
        1. Uncertainty of energy
        2. Uncertainty of force
        3. Canonical ensemble (Total energy)

    Parameters:
    
    al_type: str
        Type of active learning; 'energy', 'force', 'force_max'
    uncert_type: str
        Type of uncertainty; 'absolute', 'relative'
    uncert_shift: float
        Shifting of erf function
        (Value is relative to standard deviation)
    uncert_grad: float
        Gradient of erf function
        (Value is relative to standard deviation)
    kB: float
        Boltzmann constant in units of eV/K
    NumAtoms: int
        The number of atoms in the simulation cell
    temperature: float
        The desired temperature in units of Kelvin (K)

    Epot_step: float
        Averged of predicted potential energies at current step
    criteria_Epot_step_avg: float
        Average of potential energies
    criteria_Epot_step_std: float
        Standard deviation of potential energies

    UncertAbs_E: float or str
        Absolute uncertainty of predicted energy at current step
    criteria_UncertAbs_E_avg: float
        Average of absolute uncertainty of energies
    criteria_UncertAbs_E_std: float
        Standard deviation of absolute uncertainty of energies

    UncertRel_E: float or str
        Relative uncertainty of predicted energy at current step
    criteria_UncertRel_E_avg: float
        Average of relative uncertainty of energies
    criteria_UncertRel_E_std: float
        Standard deviation of relative uncertainty of energies

    UncertAbs_F: float or str
        Absolute uncertainty of predicted force at current step
    criteria_UncertAbs_F_avg: float
        Average of absolute uncertainty of forces
    criteria_UncertAbs_F_std: float
        Standard deviation of absolute uncertainty of forces

    UncertRel_F: float or str
        Relative uncertainty of predicted force at current step
    criteria_UncertRel_F_avg: float
        Average of relative uncertainty of forces
    criteria_UncertRel_F_std: float
        Standard deviation of relative uncertainty of forces

    Returns:

    criteria: float
        Acceptance criteria (0-1)
    """

    # Default probability
    criteria_Uncert_E = 1
    criteria_Uncert_F = 1
    
    # Calculate the probability based on energy, force, or both energy and force
    if al_type == 'energy':
        criteria_Uncert_E = get_criteria_uncert(
            uncert_type, uncert_shift, uncert_grad,
            UncertAbs_E, criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std,
            UncertRel_E, criteria_UncertRel_E_avg, criteria_UncertRel_E_std
            )
    elif al_type == 'force':
        criteria_Uncert_F = get_criteria_uncert(
            uncert_type, uncert_shift, uncert_grad,
            UncertAbs_F, criteria_UncertAbs_F_avg, criteria_UncertAbs_F_std,
            UncertRel_F, criteria_UncertRel_F_avg, criteria_UncertRel_F_std
            )
    elif al_type == 'force_max':
        # Follow the crietria proposed
        # in Y. Zhang et al. Comput. Phys. Commun. 253. 107206 (2020)
        criteria_Uncert_F = 1 if 0.05 < UncertAbs_F < 0.20 else 0
    elif al_type == 'EandFmax' or al_type == 'EorFmax': ##!! Need to be fixed.
        criteria_Uncert_E = get_criteria_uncert(
            uncert_type, uncert_shift, uncert_grad,
            UncertAbs_E, criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std,
            UncertRel_E, criteria_UncertRel_E_avg, criteria_UncertRel_E_std
            )
        criteria_Uncert_F = 1 if 0.05 < UncertAbs_F < 0.20 else 0
    else:
        single_print('You need to assign al_type.')

    # Caculate the canonical ensemble propbability using the total energy
    Prob = np.exp((-1) * (Epot_step / NumAtoms) / (kB * temperature))
    Prob_upper_limit = np.exp(
        (-1) * (criteria_Epot_step_avg / NumAtoms) / 
        (kB * temperature)
        )
    Prob_lower_limit = np.exp(
        (-1) * ((criteria_Epot_step_avg + criteria_Epot_step_std) / NumAtoms) /
        (kB * temperature)
        )

    # Get relative probability of the canomical ensemble
    criteria_Prob_inter = Prob / Prob_upper_limit;
    criteria_Prob = criteria_Prob_inter ** (
        np.log(0.2) / np.log(Prob_lower_limit / Prob_upper_limit)
        )
    # It can go beyond 1, adjust the value.
    if criteria_Prob > 1: criteria_Prob = 1;
    sys.stdout.flush()
    
    # Combine three parts of probabilities
    if al_type == 'EorFmax':
        return 1 - (1-criteria_Uncert_E) * (1-criteria_Uncert_F) * criteria_Prob
    else:
        return criteria_Uncert_E * criteria_Uncert_F * criteria_Prob
    


def get_criteria_uncert(
    uncert_type, uncert_shift, uncert_grad,
    UncertAbs, criteria_UncertAbs_avg, criteria_UncertAbs_std,
    UncertRel, criteria_UncertRel_avg, criteria_UncertRel_std
):
    """Function [get_criteria_uncert]
    Calculate a propability
    based on the average and standard deviation
    of absolute or relative uncertainty
    using the cumulative distribution function

    Parameters:
    
    uncert_type: str
        Type of uncertainty; 'absolute', 'relative'
    uncert_shift: float
        Shifting of erf function
        (Value is relative to standard deviation)
    uncert_grad: float
        Gradient of erf function
        (Value is relative to standard deviation)
    UncertAbs: float or str
        Absolute uncertainty at current step
    criteria_UncertAbs_avg: float
        Average of absolute uncertainty
    criteria_UncertAbs_std: float
        Standard deviation of absolute uncertainty

    UncertRel: float or str
        Relative uncertainty at current step
    criteria_UncertRel_avg: float
        Average of relative uncertainty
    criteria_UncertRel_std: float
        Standard deviation of relative uncertainty

    Returns:

    criteria_Uncert: float
        Probability from uncertainty values
    """
    if uncert_type == 'relative':
        criteria_Uncert = 0.5 * (
            1 + special.erf(
                (
                    (UncertRel - criteria_UncertRel_avg) -
                    uncert_shift * criteria_UncertRel_std
                ) / (uncert_grad * criteria_UncertRel_std * np.sqrt(2 * 0.1))
            )
        )
    elif uncert_type == 'absolute':
        criteria_Uncert = 0.5 * (
            1 + special.erf(
                (
                    (UncertAbs - criteria_UncertAbs_avg) -
                    uncert_shift * criteria_UncertAbs_std
                ) / (uncert_grad * criteria_UncertAbs_std * np.sqrt(2 * 0.1))
            )
        )

    return criteria_Uncert