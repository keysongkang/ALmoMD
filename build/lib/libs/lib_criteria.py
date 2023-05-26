import sys
import numpy as np
import pandas as pd
from decimal import Decimal
from scipy   import special
from libs.lib_util   import single_print


def eval_uncert(
    struc_step, nstep, nmodel, E_ref, calculator, al_type, comm, size, rank
):
    if al_type == 'energy':
        Epot_step_avg, Epot_step_std, Etot_step_avg = eval_uncert_E(struc_step, nstep, nmodel, E_ref, calculator, al_type, comm, size, rank)
        
        return Epot_step_std, Epot_step_std/Epot_step_avg, '----          ', '----          ', Etot_step_avg
    
    elif al_type == 'force':
        F_step_norm_avg, F_step_norm_std, Etot_step_avg = eval_uncert_F(struc_step, nstep, nmodel, E_ref, calculator, al_type, comm, size, rank)
        
        return '----          ', '----          ', np.average(F_step_norm_std), np.average(F_step_norm_std/F_step_norm_avg), Etot_step_avg
    
    elif al_type == 'force_max':
        F_step_norm_avg, F_step_norm_std, Etot_step_avg = eval_uncert_F(struc_step, nstep, nmodel, E_ref, calculator, al_type, comm, size, rank)
        
        return '----          ', '----          ', np.ndarray.max(F_step_norm_std), np.ndarray.max(np.array([std/avg for avg, std in zip(F_step_norm_avg, F_step_norm_std) if avg > 0.05])), Etot_step_avg
    
    elif al_type == 'EandFmax' or al_type == 'EorFmax':
        Epot_step_avg, Epot_step_std, Etot_step_avg = eval_uncert_E(struc_step, nstep, nmodel, E_ref, calculator, al_type, comm, size, rank)
        F_step_norm_avg, F_step_norm_std, Etot_step_avg = eval_uncert_F(struc_step, nstep, nmodel, E_ref, calculator, al_type, comm, size, rank)
        
        return Epot_step_std, Epot_step_std/Epot_step_avg,\
    np.ndarray.max(F_step_norm_std), np.ndarray.max(np.array([std/avg for avg, std in zip(F_step_norm_avg, F_step_norm_std) if avg > 0.05])), Etot_step_avg
    
    else:
        sys.exit("You need to set al_type.")
        
def eval_uncert_E(
    struc_step, nstep, nmodel, E_ref, calculator, al_type, comm, size, rank
):
    Epot_step = []
    Etot_step = []
    zndex = 0
    for index_nmodel in range(nmodel):
        for index_nstep in range(nstep):
            if (index_nmodel*nstep + index_nstep) % size == rank:
                struc_step.calc = calculator[zndex]
                Epot_step.append(struc_step.get_potential_energy() - E_ref)
                Etot_step.append(struc_step.get_total_energy() - E_ref)
                zndex += 1
    Epot_step = comm.allgather(Epot_step)
    Etot_step = comm.allgather(Etot_step)

    Epot_step_avg =\
    np.average(np.array([i for items in Epot_step for i in items]), axis=0)
    Epot_step_std =\
    np.std(np.array([i for items in Epot_step for i in items]), axis=0)
    Etot_step_avg =\
    np.average(np.array([i for items in Etot_step for i in items]), axis=0)
    
    return Epot_step_avg, Epot_step_std, Etot_step_avg


def eval_uncert_F(
    struc_step, nstep, nmodel, E_ref, calculator, al_type, comm, size, rank
):
    Etot_step = []
    F_step = []
    zndex = 0
    for index_nmodel in range(nmodel):
        for index_nstep in range(nstep):
            if (index_nmodel*nstep + index_nstep) % size == rank:
                struc_step.calc = calculator[zndex]
                F_step.append(struc_step.get_forces())
                Etot_step.append(struc_step.get_total_energy() - E_ref)
                zndex += 1
    F_step = comm.allgather(F_step)
    Etot_step = comm.allgather(Etot_step)

    F_step_filtered = np.array([i for items in F_step for i in items])
    F_step_avg = np.average(F_step_filtered, axis=0)
    F_step_norm = np.linalg.norm(F_step_filtered - F_step_avg, axis=1)
    F_step_norm_std = np.sqrt(np.average(F_step_norm ** 2, axis=0))
    F_step_norm_avg = np.linalg.norm(F_step_avg, axis=0)

    Etot_step_avg =\
    np.average(np.array([i for items in Etot_step for i in items]), axis=0)
    
    return F_step_norm_avg, F_step_norm_std, Etot_step_avg


def get_criteria(
    temperature, pressure, index, steps_init, size, rank
):
    uncert_data =\
    pd.read_csv(f'uncertainty-{temperature}K-{pressure}bar_{index}.txt',\
                index_col=False, delimiter='\t')
    UncerAbs_E_list = uncert_data.loc[:steps_init, 'UncertAbs_E'].values
    UncerRel_E_list = uncert_data.loc[:steps_init, 'UncertRel_E'].values
    UncerAbs_F_list = uncert_data.loc[:steps_init, 'UncertAbs_F'].values
    UncerRel_F_list = uncert_data.loc[:steps_init, 'UncertRel_F'].values
    Etot_step_list = uncert_data.loc[:steps_init, 'E_average'].values
    del uncert_data
    
    criteria_UncertAbs_E_avg = uncert_average(UncerAbs_E_list)
    criteria_UncertAbs_E_std = uncert_std(UncerAbs_E_list)
    criteria_UncertRel_E_avg = uncert_average(UncerRel_E_list)
    criteria_UncertRel_E_std = uncert_std(UncerRel_E_list)
    criteria_UncertAbs_F_avg = uncert_average(UncerAbs_F_list)
    criteria_UncertAbs_F_std = uncert_std(UncerAbs_F_list)
    criteria_UncertRel_F_avg = uncert_average(UncerRel_F_list)
    criteria_UncertRel_F_std = uncert_std(UncerRel_F_list)
    criteria_Etot_step_avg = np.average(Etot_step_list)
    criteria_Etot_step_std = np.std(Etot_step_list)
    
    if rank == 0:
        with open('result.txt', 'a') as criteriafile:
            criteriafile.write(
                f'{temperature}\t{index}\t' +
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
        criteria_Etot_step_avg, criteria_Etot_step_std
    )


def get_average(temperature, pressure, index, steps_init):
    uncert_data = pd.read_csv(f'uncertainty-{temperature}K-{pressure}bar_{index}.txt',\
                  index_col=False, delimiter='\t')
    UncerAbs_E_list = uncert_data.loc[:steps_init, 'UncertAbs_E'].values
    UncerRel_E_list = uncert_data['UncertRel_E'].values
    UncerAbs_F_list = uncert_data.loc[:steps_init, 'UncertAbs_F'].values
    UncerRel_F_list = uncert_data['UncertRel_F'].values
    
    criteria_UncertAbs_E_avg = uncert_average(UncerAbs_E_list)
    criteria_UncertRel_E_avg = uncert_average(UncerRel_E_list)
    criteria_UncertAbs_F_avg = uncert_average(UncerAbs_F_list)
    criteria_UncertRel_F_avg = uncert_average(UncerRel_F_list)

    with open('result.txt', 'a') as criteriafile:
        criteriafile.write(
            uncert_strconvter(criteria_UncertRel_E_avg) + '\t' +
            uncert_strconvter(criteria_UncertAbs_E_avg) + '\t' +
            uncert_strconvter(criteria_UncertRel_F_avg) + '\t' +
            uncert_strconvter(criteria_UncertAbs_F_avg) + '\n'
        )
        
    single_print(
        f'Average E and F uncertainty of the iteration {index} at {temperature}K'
        + f': {criteria_UncertRel_E_avg}, {criteria_UncertRel_F_avg}'
    )
    
    
def get_result(temperature, pressure, index, steps_init):
    uncert_data = pd.read_csv(f'uncertainty-{temperature}K-{pressure}bar_{index}.txt',\
                              index_col=False, delimiter='\t')
    UncerAbs_E_list = uncert_data.loc[:steps_init,'UncertAbs_E'].values
    UncerRel_E_list = uncert_data.loc[:,'UncertRel_E'].values
    UncerAbs_F_list = uncert_data.loc[:steps_init,'UncertAbs_F'].values
    UncerRel_F_list = uncert_data.loc[:,'UncertRel_F'].values
    
    criteria_UncertAbs_E_avg_all = uncert_average(UncerAbs_E_list[:])
    criteria_UncertRel_E_avg_all = uncert_average(UncerRel_E_list[:])
    criteria_UncertAbs_E_avg     = uncert_average(UncerAbs_E_list[:steps_init])
    criteria_UncertRel_E_avg     = uncert_average(UncerRel_E_list[:steps_init])
    criteria_UncertAbs_F_avg_all = uncert_average(UncerAbs_F_list[:])
    criteria_UncertRel_F_avg_all = uncert_average(UncerRel_F_list[:])
    criteria_UncertAbs_F_avg     = uncert_average(UncerAbs_F_list[:steps_init])
    criteria_UncertRel_F_avg     = uncert_average(UncerRel_F_list[:steps_init])

    with open('result.txt', 'a') as criteriafile:
        criteriafile.write(
            f'{temperature}\t{index}\t' +
            uncert_strconvter(criteria_UncertRel_E_avg) + '\t' +
            uncert_strconvter(criteria_UncertAbs_E_avg) + '\t' +
            uncert_strconvter(criteria_UncertRel_F_avg) + '\t' +
            uncert_strconvter(criteria_UncertAbs_F_avg) + '\t' +
            uncert_strconvter(criteria_UncertRel_E_avg_all) + '\t' +
            uncert_strconvter(criteria_UncertAbs_E_avg_all) + '\t' +
            uncert_strconvter(criteria_UncertRel_F_avg_all) + '\t' +
            uncert_strconvter(criteria_UncertAbs_F_avg_all) + '\n'
        )

    
def uncert_average(itemlist):
    return '----          ' if itemlist[0] == '----          ' else np.average(itemlist)
    
    
def uncert_std(itemlist):
    return '----          ' if itemlist[0] == '----          ' else np.std(itemlist)

def uncert_strconvter(value):
    if value == '----          ':
        return value
    return '{:.5e}'.format(Decimal(value))
    
    
def get_criteria_prob(
    al_type, uncert_type, kB, NumAtoms, temperature, Etot_step, criteria_Etot_step_avg, criteria_Etot_step_std,
    UncertAbs_E, criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std, UncertRel_E, criteria_UncertRel_E_avg, criteria_UncertRel_E_std,
    UncertAbs_F, criteria_UncertAbs_F_avg, criteria_UncertAbs_F_std, UncertRel_F, criteria_UncertRel_F_avg, criteria_UncertRel_F_std
):
    criteria_Uncert_E = 1
    criteria_Uncert_F = 1
    
    if al_type == 'energy':
        criteria_Uncert_E = get_criteria_uncert(uncert_type, UncertAbs_E, criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std, UncertRel_E, criteria_UncertRel_E_avg, criteria_UncertRel_E_std)
    elif al_type == 'force':
        criteria_Uncert_F = get_criteria_uncert(uncert_type, UncertAbs_F, criteria_UncertAbs_F_avg, criteria_UncertAbs_F_std, UncertRel_F, criteria_UncertRel_F_avg, criteria_UncertRel_F_std)
    elif al_type == 'force_max':
        criteria_Uncert_F = 1 if 0.05 < UncertAbs_F < 0.20 else 0
    elif al_type == 'EandFmax' or al_type == 'EorFmax':
        criteria_Uncert_E = get_criteria_uncert(uncert_type, UncertAbs_E, criteria_UncertAbs_E_avg, criteria_UncertAbs_E_std, UncertRel_E, criteria_UncertRel_E_avg, criteria_UncertRel_E_std)
        criteria_Uncert_F = 1 if 0.05 < UncertAbs_F < 0.20 else 0
    else:
        "Nothing"

    Prob = np.exp((-1) * (Etot_step / NumAtoms) / (kB * temperature))
    Prob_upper_limit =\
    np.exp((-1) * (criteria_Etot_step_avg / NumAtoms) / (kB * temperature))
    Prob_lower_limit =\
    np.exp((-1) * ((criteria_Etot_step_avg+criteria_Etot_step_std) / NumAtoms) / (kB * temperature))

    criteria_Prob_inter = Prob / Prob_upper_limit;
    criteria_Prob = criteria_Prob_inter ** (np.log(0.2)/np.log(Prob_lower_limit/Prob_upper_limit))
    if criteria_Prob > 1: criteria_Prob = 1;
    sys.stdout.flush()
    
    if al_type == 'EorFmax':
        return 1 - (1-criteria_Uncert_E) * (1-criteria_Uncert_F) * criteria_Prob
    else:
        return criteria_Uncert_E * criteria_Uncert_F * criteria_Prob
    

def get_criteria_uncert(
    uncert_type, UncertAbs, criteria_UncertAbs_avg, criteria_UncertAbs_std,
    UncertRel, criteria_UncertRel_avg, criteria_UncertRel_std
):
    if uncert_type == 'relative':
        criteria_Uncert = 0.5 *\
        (1 + special.erf(((UncertRel-criteria_UncertRel_avg)-\
                          0.2661*criteria_UncertRel_std)/(criteria_UncertRel_std*np.sqrt(2*0.1))))
        
    elif uncert_type == 'absolute':
        criteria_Uncert = 0.5 *\
        (1 + special.erf(((UncertAbs-criteria_UncertAbs_avg)-\
                          0.2661*criteria_UncertAbs_std)/(criteria_UncertAbs_std*np.sqrt(2*0.1))))
        
    return criteria_Uncert