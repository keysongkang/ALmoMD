import os
import sys
import numpy as np
import pandas as pd

from decimal import Decimal
from nequip.ase import nequip_calculator

from ase import Atoms

import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import r2_score, mean_absolute_error

from libs.lib_util  import output_init, single_print, check_mkdir

import torch
torch.set_default_dtype(torch.float64)


def run_dft_test(inputs):
    """Function [run_dft_test]
    Check the validation error.
    """

    output_init('test', inputs.version)
    single_print(f'[test]\tInitiate the validation test process')

    # Read testing data
    npz_test = f'./MODEL/data-test.npz' # Path
    data_test = np.load(npz_test)         # Load
    single_print(f'[test]\tRead the testing data: data-test.npz')

    single_print(f'\t\tDevice: {inputs.device}')

    for test_idx in range(inputs.test_index):
        # Set the path to folders storing the training data for NequIP
        workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{test_idx}'
        # Initialization of a termination signal
        signal = 0

        single_print(f'[test]\tFind the trained models: {workpath}')
        # Load the trained models as calculators
        calc_MLIP = []
        for index_nmodel in range(inputs.nmodel):
            for index_nstep in range(inputs.nstep):
                dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
                if os.path.exists(f'{workpath}/{dply_model}'):
                    single_print(f'\t\tFound the deployed model: {dply_model}')
                    calc_MLIP.append(
                        nequip_calculator.NequIPCalculator.from_deployed_model(
                            f'{workpath}/{dply_model}', device=inputs.device
                            )
                    )
                else:
                    # If there is no model, turn on the termination signal
                    single_print(f'\t\tCannot find the model: {dply_model}')
                    signal = 1

        # Check the termination signal
        if signal == 1:
            single_print('[test]\tSome training processes are not finished.')
            sys.exit()

        # Open the file to store the results
        outputfile = open(f'result-test_{test_idx}_energy.txt', 'w')
        outputfile.write('index   \tUncertAbs\tRealErrorAbs\tRealE\tPredictE\n')
        outputfile.close()
        
        single_print(f'[test]\tGo through all configurations in the testing data ...')
        # Go through all configurations in the testing data
        config_idx = 1
        for id_E, id_F, id_R, id_z, id_CELL, id_PBC in zip(
            data_test['E'], data_test['F'], data_test['R'],
            data_test['z'], data_test['CELL'], data_test['PBC']
            ):
            # Create the corresponding ASE atoms
            struc = Atoms(id_z, positions=id_R, cell=id_CELL, pbc=id_PBC)
            # Prepare the empty lists for predicted energy and force
            prd_E = []
            prd_F = []
            zndex = 0

            # Go through all trained models
            for index_nmodel in range(inputs.nmodel):
                for index_nstep in range(inputs.nstep):
                    struc.calc = calc_MLIP[zndex]
                    prd_E.append(struc.get_potential_energy())
                    prd_F.append(struc.get_forces())
                    zndex += 1

            # Get average and standard deviation (Uncertainty) of predicted energies from various models
            prd_E_avg = np.average(prd_E)
            prd_E_std = np.std(prd_E)

            # Get the real error
            realerror_E = np.absolute(prd_E_avg - id_E)

            # Get average of predicted forces from various models
            if inputs.harmonic_F:
                from libs.lib_util import get_fc_ha
                F_ha = get_fc_ha(struc.get_positions(), 'geometry.in.supercell', 'FORCE_CONSTANTS_remapped')
                prd_F = prd_F + F_ha

            prd_F_avg = np.average(prd_F, axis=0)

            # Save all energy information
            trajfile = open(f'result-test_{test_idx}_energy.txt', 'a')
            trajfile.write(
                str(config_idx) + '          \t' +
                '{:.5e}'.format(Decimal(str(prd_E_std))) + '\t' +       # Standard deviation (Uncertainty)
                '{:.5e}'.format(Decimal(str(realerror_E))) + '\t' +     # Real error
                '{:.10e}'.format(Decimal(str(id_E))) + '\t' +           # Real energy
                '{:.10e}'.format(Decimal(str(prd_E_avg))) + '\n'        # Predicted energy
            )
            trajfile.close()

            # Save all force information
            trajfile = open(f'result-test_{test_idx}_force.txt', 'a')
            for kndex in range(len(prd_F_avg)):
                for lndex in range(3):
                    trajfile.write(
                        '{:.10e}'.format(Decimal(str(id_F[kndex][lndex]))) + '\t' +     # Real force
                        '{:.10e}'.format(Decimal(str(prd_F_avg[kndex][lndex]))) + '\n'  # Predicted force
                    )
            trajfile.close()

            config_idx += 1

        single_print(f'[test]\tPlot the results ...')
        ## Plot the energy and force prediction results
        # Read the energy data
        data = pd.read_csv(f'result-test_{test_idx}_energy.txt', sep="\t")

        # Font style and font size
        plt.rcParams['font.sans-serif'] = "Helvetica"
        plt.rcParams['font.size'] = "23"

        # Prepare subplots
        fig, ax1 = plt.subplots(figsize=(6.5, 6), dpi=80)

        # Plot data
        ax1.plot(data['RealE'], data['PredictE'], color="orange", linestyle='None', marker='o')
        ax1.plot([0, 1], [0, 1], transform=ax1.transAxes)

        # Axis information
        ax1.set_xlabel('Real E')
        ax1.set_ylabel('Predicted E')
        ax1.tick_params(direction='in')
        plt.tight_layout()
        plt.show()
        fig.savefig('figure_energy.png') # Save the figure

        # Get the energy statistics
        R2_E = r2_score(data['RealE'], data['PredictE'])
        MAE_E = mean_absolute_error(data['RealE'], data['PredictE'])

        # Read the force data
        data = pd.read_csv(f'result-test_{test_idx}_force.txt', sep="\t", header=None)

        # Prepare subplots
        fig, ax1 = plt.subplots(figsize=(6.5, 6), dpi=80)

        # Plot data
        ax1.plot(data[0], data[1], color="orange", linestyle='None', marker='o')
        ax1.plot([0, 1], [0, 1], transform=ax1.transAxes)

        # Axis information
        ax1.set_xlabel('Real F')
        ax1.set_ylabel('Predicted F')
        ax1.tick_params(direction='in')
        plt.tight_layout()
        plt.show()
        fig.savefig('figure_force.png') # Save the figure

        # Get the force statistics
        R2_F = r2_score(data[0], data[1])
        MAE_F = mean_absolute_error(data[0], data[1])

        # Print out the statistic results
        single_print(f'[test]\t[[Statistic results]]')
        single_print(f'[test]\tEnergy_R2\tEnergy_MAE\tForces_R2\tForces_MAE')
        single_print(f'[test]\t{R2_E}\t{MAE_E}\t{R2_F}\t{MAE_F}')

    single_print(f'[test]\t!! Finish the testing process')
