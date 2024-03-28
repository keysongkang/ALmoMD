import os
import sys
import numpy as np
import pandas as pd
from decimal import Decimal
from scipy import special
from libs.lib_util import single_print


def eval_uncert(
    struc_step, nstep, nmodel, E_ref, calculator, al_type, harmonic_F
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

    Epot_step_avg, Epot_step_std, F_step_norm_avg, F_step_norm_std, S_step_avg, S_step_std\
    = eval_uncert_all(struc_step, nstep, nmodel, E_ref, calculator, al_type, harmonic_F)

    from libs.lib_util import empty_inputs
    uncerts = empty_inputs()

    if al_type == 'energy' or al_type == 'force' or al_type == 'sigma' or al_type == 'all':
        uncerts.UncertAbs_E = Epot_step_std
        uncerts.UncertRel_E = Epot_step_std / Epot_step_avg
        uncerts.UncertAbs_F = np.average(F_step_norm_std)
        uncerts.UncertRel_F = np.average(F_step_norm_std / F_step_norm_avg)
        uncerts.UncertAbs_S = S_step_std
        uncerts.UncertRel_S = S_step_std / S_step_avg

        return (uncerts, Epot_step_avg, S_step_avg)

    # Active learning based on the MAXIUM uncertainty of predicted force
    elif al_type == 'force_max':
        uncerts.UncertAbs_E = Epot_step_std
        uncerts.UncertRel_E = Epot_step_std / Epot_step_avg
        uncerts.UncertAbs_F = np.ndarray.max(F_step_norm_std)
        uncerts.UncertRel_F = np.ndarray.max(
            np.array([std / avg for avg, std in zip(F_step_norm_avg, F_step_norm_std)])
            )
        uncerts.UncertAbs_S = S_step_std
        uncerts.UncertRel_S = S_step_std / S_step_avg

        return (uncerts, Epot_step_avg, S_step_avg)

    else:
        sys.exit("You need to set al_type.")
        

def eval_uncert_all(
    struc_step, nstep, nmodel, E_ref, calculator, al_type, harmonic_F
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

    from mpi4py import MPI
    from libs.lib_util import eval_sigma

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Prepare empty lists for potential and total energies
    Epot_step = []
    F_step = []
    prd_struc = []
    zndex = 0

    # Get predicted potential and total energies shifted by E_ref (ground state energy)
    for index_nmodel in range(nmodel):
        for index_nstep in range(nstep):
            if (index_nmodel*nstep + index_nstep) % size == rank:
                struc_step.calc = calculator[zndex]
                Epot_step.append(struc_step.get_potential_energy() - E_ref)
                F_step.append(struc_step.get_forces())
                prd_struc.append(struc_step.get_positions())
                zndex += 1
    Epot_step = comm.allgather(Epot_step)
    F_step = comm.allgather(F_step)
    prd_struc = comm.allgather(prd_struc) 

    # Get the average and standard deviation of predicted potential energies
    # and the average of total energies
    Epot_step_filtered = np.array([jtem for item in Epot_step if len(item) != 0 for jtem in item])
    # Get the average and standard deviation of the norm of predicted forces
    F_step_filtered = np.array([jtem for item in F_step if len(item) != 0 for jtem in item])
    struc_filtered = [jtem for item in prd_struc if len(item) != 0 for jtem in item]

    if harmonic_F:
        from libs.lib_util import get_displacements, get_fc_ha, get_E_ha
        displacements = get_displacements(struc_step.get_positions(), 'geometry.in.supercell')
        F_ha = get_fc_ha(displacements, 'FORCE_CONSTANTS_remapped')
        E_ha = get_E_ha(displacements, F_ha)
        Epot_step_filtered = Epot_step_filtered + E_ha
        F_step_filtered = F_step_filtered + F_ha

    Epot_step_avg = np.average(Epot_step_filtered, axis=0)
    Epot_step_std = np.std(Epot_step_filtered, axis=0)

    F_step_avg = np.average(F_step_filtered, axis=0)
    F_step_norm = np.array([[np.linalg.norm(Fcomp) for Fcomp in Ftems] for Ftems in F_step_filtered - F_step_avg])
    F_step_norm_std = np.sqrt(np.average(F_step_norm ** 2, axis=0))
    F_step_norm_avg = np.linalg.norm(F_step_avg, axis=1)

    prd_sigma = []
    for prd_F_step, prd_struc_step in zip(F_step_filtered, struc_filtered):
        prd_sigma.append(eval_sigma(prd_F_step, prd_struc_step, al_type))

    # Get the average and standard deviation of the norm of predicted forces
    sigma_step_avg = np.average(prd_sigma, axis=0)
    sigma_step_std = np.std(prd_sigma, axis=0)

    return Epot_step_avg, Epot_step_std, F_step_norm_avg, F_step_norm_std, sigma_step_avg, sigma_step_std


def get_criteria(
    temperature, pressure, index, steps_init, al_type
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

    from libs.lib_util import empty_inputs
    criteria = empty_inputs()

    # Read all uncertainty results
    result_data = pd.read_csv(
        f'result.txt',
        index_col=False, delimiter='\t'
        )

    # if ensemble == 'NVTLangevin_meta':
    #     # Get their average and standard deviation
    #     criteria.Epotential_avg = result_data.loc[:, 'E_potent_avg_i'].to_numpy()[0]
    #     criteria.Epotential_std = result_data.loc[:, 'E_potent_std_i'].to_numpy()[0]
    # else:
    #     # Get their average and standard deviation
    criteria.Epotential_avg = result_data.loc[:, 'E_potent_avg_i'].to_numpy()[-1]
    criteria.Epotential_std = result_data.loc[:, 'E_potent_std_i'].to_numpy()[-1]
    
    if al_type == 'energy':
        criteria.Un_Abs_E_avg_i = result_data.loc[:, 'Un_Abs_E_avg_i'].to_numpy()[-1]
        criteria.Un_Abs_E_std_i = result_data.loc[:, 'Un_Abs_E_std_i'].to_numpy()[-1]
        criteria.Un_Rel_E_avg_i = result_data.loc[:, 'Un_Rel_E_avg_i'].to_numpy()[-1]
        criteria.Un_Rel_E_std_i = result_data.loc[:, 'Un_Rel_E_std_i'].to_numpy()[-1]
    else:
        criteria.Un_Abs_E_avg_i = 0.0
        criteria.Un_Abs_E_std_i = 0.0
        criteria.Un_Rel_E_avg_i = 0.0
        criteria.Un_Rel_E_std_i = 0.0

    if al_type == 'force' or al_type == 'force_max':
        criteria.Un_Abs_F_avg_i = result_data.loc[:, 'Un_Abs_F_avg_i'].to_numpy()[-1]
        criteria.Un_Abs_F_std_i = result_data.loc[:, 'Un_Abs_F_std_i'].to_numpy()[-1]
        criteria.Un_Rel_F_avg_i = result_data.loc[:, 'Un_Rel_F_avg_i'].to_numpy()[-1]
        criteria.Un_Rel_F_std_i = result_data.loc[:, 'Un_Rel_F_std_i'].to_numpy()[-1]
    else:
        criteria.Un_Abs_F_avg_i = 0.0
        criteria.Un_Abs_F_std_i = 0.0
        criteria.Un_Rel_F_avg_i = 0.0
        criteria.Un_Rel_F_std_i = 0.0

    if al_type == 'sigma' or al_type == 'sigma_max':
        criteria.Un_Abs_S_avg_i = result_data.loc[:, 'Un_Abs_S_avg_i'].to_numpy()[-1]
        criteria.Un_Abs_S_std_i = result_data.loc[:, 'Un_Abs_S_std_i'].to_numpy()[-1]
        criteria.Un_Rel_S_avg_i = result_data.loc[:, 'Un_Rel_S_avg_i'].to_numpy()[-1]
        criteria.Un_Rel_S_std_i = result_data.loc[:, 'Un_Rel_S_std_i'].to_numpy()[-1]
    else:
        criteria.Un_Abs_S_avg_i = 0.0
        criteria.Un_Abs_S_std_i = 0.0
        criteria.Un_Rel_S_avg_i = 0.0
        criteria.Un_Rel_S_std_i = 0.0

    return criteria


def get_result(inputs, get_type):
    """Function [get_result]
    Get average and standard deviation of absolute and relative undertainty
    of energies and forces and also those of total energy for all steps

    Parameters:

    inputs.temperature: float
        The desired temperature in units of Kelvin (K)
    inputs.pressure: float
        The desired pressure in units of eV/Angstrom**3
    inputs.index: int
        The index of AL interactive step
    inputs.steps_init: int
        Initialize MD steps to get averaged uncertainties and energies
    """

    if get_type == 'progress':
        get_index = inputs.index-1
    else:
        get_index = inputs.index

    # Read all uncertainty results
    uncert_data = pd.read_csv(
        f'UNCERT/uncertainty-{inputs.temperature}K-{inputs.pressure}bar_{get_index}.txt',
        index_col=False, delimiter='\t'
        )

    result_print = ''
    # Get their average and standard deviation
    if inputs.al_type == 'energy':
        UncerAbs_E_list = uncert_data.loc[:,'UncertAbs_E'].values
        UncerRel_E_list = uncert_data.loc[:,'UncertRel_E'].values
        criteria_UncertAbs_E_avg_all = uncert_average(UncerAbs_E_list[:])
        criteria_UncertRel_E_avg_all = uncert_average(UncerRel_E_list[:])
        result_print +=   '\t' + uncert_strconvter(criteria_UncertRel_E_avg_all)\
                        + '\t' + uncert_strconvter(criteria_UncertAbs_E_avg_all)

    if inputs.al_type == 'force' or inputs.al_type == 'force_max':
        UncerAbs_F_list = uncert_data.loc[:,'UncertAbs_F'].values
        UncerRel_F_list = uncert_data.loc[:,'UncertRel_F'].values
        criteria_UncertAbs_F_avg_all = uncert_average(UncerAbs_F_list[:])
        criteria_UncertRel_F_avg_all = uncert_average(UncerRel_F_list[:])
        result_print +=   '\t' + uncert_strconvter(criteria_UncertRel_F_avg_all)\
                        + '\t' + uncert_strconvter(criteria_UncertAbs_F_avg_all)

    if inputs.al_type == 'sigma' or inputs.al_type == 'sigma_max':
        UncerAbs_S_list = uncert_data.loc[:,'UncertAbs_S'].values
        UncerRel_S_list = uncert_data.loc[:,'UncertRel_S'].values
        criteria_UncertAbs_S_avg_all = uncert_average(UncerAbs_S_list[:])
        criteria_UncertRel_S_avg_all = uncert_average(UncerRel_S_list[:])
        result_print +=   '\t' + uncert_strconvter(criteria_UncertAbs_S_avg_all)\
                        + '\t' + uncert_strconvter(criteria_UncertRel_S_avg_all)

    # Record the average values
    if inputs.rank == 0:
        with open('result.txt', 'a') as criteriafile:
            criteriafile.write(result_print+ '\n')


    
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
    

def get_criteria_prob(inputs, Epot_step, uncerts, criteria):
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
    criteria_Uncert_S = 1

    # Calculate the probability based on energy, force, or both energy and force
    if inputs.al_type == 'energy':
        criteria_Uncert_E = get_criteria_uncert(
            inputs.uncert_type, inputs.uncert_shift, inputs.uncert_grad,
            uncerts.UncertAbs_E, criteria.Un_Abs_E_avg_i, criteria.Un_Abs_E_std_i,
            uncerts.UncertRel_E, criteria.Un_Rel_E_avg_i, criteria.Un_Rel_E_std_i
            )
    elif inputs.al_type == 'force' or inputs.al_type == 'force_max':
        criteria_Uncert_F = get_criteria_uncert(
            inputs.uncert_type, inputs.uncert_shift, inputs.uncert_grad,
            uncerts.UncertAbs_F, criteria.Un_Abs_F_avg_i, criteria.Un_Abs_F_std_i,
            uncerts.UncertRel_F, criteria.Un_Rel_F_avg_i, criteria.Un_Rel_F_std_i
            )
    elif inputs.al_type == 'sigma':
        criteria_Uncert_S = get_criteria_uncert(
            inputs.uncert_type, inputs.uncert_shift, inputs.uncert_grad,
            uncerts.UncertAbs_S, criteria.Un_Abs_S_avg_i, criteria.Un_Abs_S_std_i,
            uncerts.UncertRel_S, criteria.Un_Rel_S_avg_i, criteria.Un_Rel_S_std_i
            )
    # elif al_type == 'force_max':
    #     # Follow the crietria proposed
    #     # in Y. Zhang et al. Comput. Phys. Commun. 253. 107206 (2020)
    #     criteria_Uncert_F = 1 if 0.05 < UncertAbs_F < 0.20 else 0
    else:
        sys.exit("You need to set al_type.")

    beta = inputs.kB * inputs.temperature

    if inputs.ensemble == 'NVTLangevin_meta' or inputs.ensemble == 'NVTLangevin_bias' or inputs.ensemble == 'NPTisoiso' :
        criteria_Prob = 1
    else:
        # Caculate the canonical ensemble propbability using the total energy
        Prob = np.exp((-1) * (Epot_step / inputs.NumAtoms) / beta)
        Prob_upper_limit = np.exp(
            (-1) * ((criteria.Epotential_avg + criteria.Epotential_std) / inputs.NumAtoms) / beta)
        Prob_lower_limit = np.exp(
            (-1) * ((criteria.Epotential_avg + criteria.Epotential_std*1.8) / inputs.NumAtoms) / beta)

        # Get relative probability of the canomical ensemble
        criteria_Prob_inter = Prob / Prob_upper_limit;
        criteria_Prob = criteria_Prob_inter ** (
            np.log(0.2) / np.log(Prob_lower_limit / Prob_upper_limit)
            )
        # It can go beyond 1, adjust the value.
        if criteria_Prob > 1: criteria_Prob = 1;
        sys.stdout.flush()

    # Combine three parts of probabilities
    if inputs.al_type == 'EorFmax':
        return 1 - (1-criteria_Uncert_E) * (1-criteria_Uncert_F) * criteria_Uncert_S * criteria_Prob
    else:
        return criteria_Uncert_E * criteria_Uncert_F * criteria_Uncert_S * criteria_Prob
    


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