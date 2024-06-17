import os
from vibes import son
import sys
import numpy as np

from nequip.ase import nequip_calculator

from ase import Atoms
from ase.build import make_supercell
from ase.data   import atomic_numbers
from ase.io import read as atoms_read
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from libs.lib_md import runMD
from libs.lib_util  import output_init, single_print, check_mkdir

import torch
torch.set_default_dtype(torch.float64)


def run_dft_runmd(inputs):
    """Function [run_dft_runmd]
    Initiate MD calculation using trained models.
    """

    # Print the head
    output_init('runMD', inputs.version)
    single_print(f'[runMD]\tInitiate runMD process')

    # Initialization of a termination signal
    signal = 0

    single_print(f'[runMD]\tCheck the initial configuration')
    if os.path.exists('start.in'):
        print(f'[runMD]\tFound the start.in file. MD starts from this.')
        # Read the ground state structure with the primitive cell
        struc_init = atoms_read('start.in', format='aims')
        # Make it supercell
        struc = make_supercell(struc_init, inputs.supercell_init)
        MaxwellBoltzmannDistribution(struc, temperature_K=inputs.temperature, force_temp=True)
    elif os.path.exists('start.traj'):
        single_print(f'[runMD]\tFound the start.traj file. MD starts from this.')
        # Read the ground state structure with the primitive cell
        struc_init = Trajectory('start.traj')[-1]
        struc = make_supercell(struc_init, inputs.supercell_init)
        del struc_init
        try:
            struc.get_velocities()
        except AttributeError:
            MaxwellBoltzmannDistribution(struc, temperature_K=inputs.temperature, force_temp=True)
    elif os.path.exists('start.bundle'):
        from ase.io.bundletrajectory import BundleTrajectory
        single_print(f'[runMD]\tFound the start.bundle file. MD starts from this.')
        file_traj_read = BundleTrajectory(filename='start.bundle', mode='r')
        file_traj_read[0]; #ASE bug
        struc_init = file_traj_read[-1]
        struc = make_supercell(struc_init, inputs.supercell_init)
        del struc_init
        try:
            struc.get_velocities()
        except AttributeError:
            MaxwellBoltzmannDistribution(struc, temperature_K=inputs.temperature, force_temp=True)
    else:
        single_print(f'[runMD]\tMD starts from the last entry of the trajectory.son file')
        # Read all structural configurations in SON file
        metadata, data = son.load('trajectory.son')
        atom_numbers = [
        atomic_numbers[items[1]]
        for items in data[-1]['atoms']['symbols']
        for jndex in range(items[0])
        ]
        struc_son = Atoms(
            atom_numbers,
            positions=data[-1]['atoms']['positions'],
            cell=data[-1]['atoms']['cell'],
            pbc=data[-1]['atoms']['pbc']
            )
        struc_son.set_velocities(data[-1]['atoms']['velocities'])
        struc = make_supercell(struc_son, inputs.supercell_init)

    ## Prepare the ground state structure
    # Read the ground state structure with the primitive cell
    struc_init = atoms_read('geometry.in.supercell', format='aims')
    single_print(f'[runMD]\tRead the reference structure: geometry.in.next_step')

    cell_new = struc.get_cell()
    struc.set_cell(cell_new * inputs.cell_factor)

    single_print(f'[runMD]\tFind the trained models: {inputs.modelpath}')
    # Prepare empty lists for potential and total energies
    Epot_step = []
    calc_MLIP = []

    single_print(f'\t\tDevice: {inputs.device}')

    for index_nmodel in range(inputs.nmodel):
        for index_nstep in range(inputs.nstep):
            dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
            if os.path.exists(f'{inputs.modelpath}/{dply_model}'):
                single_print(f'\t\tFound the deployed model: {dply_model}')
                calc_MLIP.append(
                    nequip_calculator.NequIPCalculator.from_deployed_model(
                        f'{inputs.modelpath}/{dply_model}', device=inputs.device
                        )
                )
                struc_init.calc = calc_MLIP[-1]
                Epot_step.append(struc_init.get_potential_energy() - inputs.E_gs)
            else:
                # If there is no model, turn on the termination signal
                single_print(f'\t\tCannot find the model: {dply_model}')
                signal = 1

    # Get averaged energy from trained models
    Epot_step_avg =\
    np.average(np.array([i for items in Epot_step for i in items]), axis=0)
    single_print(f'[runMD]\tGet the potential energy of the reference structure: {Epot_step_avg}')

    # if the number of trained model is not enough, terminate it
    if signal == 1:
        sys.exit()

    single_print(f'[runMD]\tInitiate MD with trained models')

    if inputs.MD_search == 'restart':
        sigma = 0.0
        stepss = 300
        while sigma < 1.0:
            single_print(f'[runMD]\tFound the start.traj file. MD starts from this.')
            # Read the ground state structure with the primitive cell
            struc_init = Trajectory('start.traj')[0]
            struc = make_supercell(struc_init, inputs.supercell_init)
            try:
                struc.get_velocities()
            except AttributeError:
                MaxwellBoltzmannDistribution(struc, temperature_K=inputs.temperature, force_temp=True)

            runMD(
                inputs, struc, stepss,
                inputs.logfile, inputs.trajectory, calc_MLIP,
                signal_uncert=inputs.signal_uncert, signal_append=False
                )

            import pandas as pd
            data = pd.read_csv('md.log', sep='\t')
            sigma = np.array(data['S_average'])[-1]
    else:
        runMD(
            inputs, struc, inputs.steps,
            inputs.logfile, inputs.trajectory, calc_MLIP,
            signal_uncert=inputs.signal_uncert, signal_append=True
            )

    # runMD(
    #     inputs, struc, inputs.steps,
    #     inputs.logfile, inputs.trajectory, calc_MLIP,
    #     signal_uncert=True, signal_append=True
    #     )

    single_print(f'[runMD]\t!! Finish MD calculations')
