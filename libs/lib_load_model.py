import os
import sys
from nequip.ase import nequip_calculator
from libs.lib_util     import mpi_print

from glp import instantiate
from glp.ase import Calculator
import torch
torch.set_default_dtype(torch.float64)


def load_model(inputs):

    # Set the path to folders storing the training data for NequIP
    workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'

    # Initialization of a termination signal
    signal = 0

    mpi_print(f'\t\tDevice: {inputs.device}', inputs.rank)

    # Load the trained models as calculators
    inputs.calc_MLIP = []
    for index_nmodel in range(inputs.nmodel):
        for index_nstep in range(inputs.nstep):
            if (index_nmodel * inputs.nstep + index_nstep) % inputs.size == inputs.rank:

                if inputs.MLIP == 'nequip':
                    dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
                    if os.path.exists(f'{workpath}/{dply_model}'):
                        mpi_print(f'\t\tFound the deployed model: {dply_model}', rank=0)
                        inputs.calc_MLIP.append(
                            nequip_calculator.NequIPCalculator.from_deployed_model(f'{workpath}/{dply_model}', device=inputs.device)
                        )
                    else:
                        # If there is no model, turn on the termination signal
                        mpi_print(f'\t\tCannot find the model: {dply_model}', rank=0)
                        signal = 1
                        signal = inputs.comm.bcast(signal, root=inputs.rank)
                elif inputs.MLIP == 'so3krates':
                    potential_dict = {"mlff": {"folder": f"{workpath}/deployed-model_{index_nmodel}_{index_nstep}"}}
                    get_calculator = instantiate.get_calculator(potential_dict, {"atom_pair": {"skin": 0.1}})
                    inputs.calc_MLIP.append(Calculator(get_calculator))

    # Check the termination signal
    if signal == 1:
        mpi_print('[Termi]\tSome training processes are not finished.', inputs.rank)
        sys.exit()

    return inputs