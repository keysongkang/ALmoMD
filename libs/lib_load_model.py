import os
import sys
from libs.lib_util     import single_print

def load_model(inputs):

    # Set the path to folders storing the training data for NequIP
    workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'

    # Initialization of a termination signal
    signal = 0

    single_print(f'\t\tDevice: {inputs.device}')

    # Load the trained models as calculators
    inputs.calc_MLIP = []
    for index_nmodel in range(inputs.nmodel):
        for index_nstep in range(inputs.nstep):

            if inputs.MLIP == 'nequip':
                from nequip.ase import nequip_calculator
                                
                import torch
                torch.set_default_dtype(torch.float64)

                dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
                if os.path.exists(f'{workpath}/{dply_model}'):
                    single_print(f'\t\tFound the deployed model: {dply_model}')
                    inputs.calc_MLIP.append(
                        nequip_calculator.NequIPCalculator.from_deployed_model(f'{workpath}/{dply_model}', device=inputs.device)
                    )
                else:
                    # If there is no model, turn on the termination signal
                    single_print(f'\t\tCannot find the model: {dply_model}')
                    signal = 1
            elif inputs.MLIP == 'so3krates':
                from glp import instantiate
                from glp.ase import Calculator
                potential_dict = {"mlff": {"folder": f"{workpath}/deployed-model_{index_nmodel}_{index_nstep}"}}
                get_calculator = instantiate.get_calculator(potential_dict, {"heat_flux_unfolded": {"skin": 1.0}})
                inputs.calc_MLIP.append(Calculator(get_calculator))

    # Check the termination signal
    if signal == 1:
        single_print('[Termi]\tSome training processes are not finished.')
        sys.exit()

    return inputs