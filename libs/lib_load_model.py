import os
import sys
import threading
import warnings
import copy

from libs.lib_util import single_print


def ensemble_calculate(calculators, atoms):
    """Run all NequIPCalculators on `atoms` in parallel threads.

    Build the neighbor list once and clone it per model; requires every
    calculator to share r_max (asserted in load_model). Each thread
    runs on its own atoms.copy() so nequip's calculate() handles result
    extraction with no race on the shared atoms object.

    On multi-GPU runs (models on cuda:0..N), forward passes run truly
    in parallel. On a single GPU kernels are still serialized by CUDA,
    but there is no regression versus a sequential loop.

    Returns a list of per-model results dicts, one per calculator in
    the same order as `calculators`. Each dict holds whatever keys the
    underlying nequip calculator populated (energy, free_energy,
    energies, forces, stress, stresses, ...).
    """
    if not calculators:
        return []

    if not hasattr(calculators[0], 'r_max'):
        results = []
        for calc in calculators:
            atoms_copy = atoms.copy()
            calc.calculate(atoms_copy)
            results.append(copy.deepcopy(calc.results))
        return results

    from nequip.data import AtomicData

    # Build the neighbor list once; every call to from_ase inside the
    # threaded calculate() invocations returns a fresh clone.
    prebuilt = AtomicData.from_ase(atoms=atoms, r_max=calculators[0].r_max)
    original = AtomicData.__dict__['from_ase']
    call_count = 0
    call_lock = threading.Lock()

    def _cached(atoms=None, r_max=None, **kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
        return prebuilt.clone()

    AtomicData.from_ase = _cached
    try:
        atoms_copies = [atoms.copy() for _ in calculators]
        errors = []

        def _calculate(calc, atoms_copy):
            try:
                calc.calculate(atoms_copy)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_calculate, args=(calc, ac))
            for calc, ac in zip(calculators, atoms_copies)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
    finally:
        AtomicData.from_ase = original

    # Warn if nequip stopped routing through from_ase upstream: the
    # neighbor-list sharing silently stops working in that case.
    if call_count < len(calculators):
        warnings.warn(
            f'ensemble_calculate: AtomicData.from_ase called {call_count} '
            f'times, expected {len(calculators)}. nequip internals may '
            f'have changed; neighbor-list sharing is no longer effective.',
            RuntimeWarning,
        )

    return [copy.deepcopy(calc.results) for calc in calculators]


def load_model(inputs):

    # Set the path to folders storing the training data for NequIP
    workpath = f'./MODEL/{inputs.temperature}K-{inputs.pressure}bar_{inputs.index}'

    # Initialization of a termination signal
    signal = 0

    single_print(f'\t\tDevice: {inputs.device}')

    # Load the trained models as calculators
    inputs.calc_MLIP = []
    if inputs.MLIP == 'nequip':
        from nequip.ase import nequip_calculator
                        
        import torch
        torch.set_default_dtype(torch.float64)

        # check cuda availability
        if torch.cuda.is_available():
            single_print("CUDA is available!")
            num_devices = torch.cuda.device_count()
            single_print(f"Number of CUDA devices: {num_devices}")

            for i in range(num_devices):
                device_name = torch.cuda.get_device_name(i)
                single_print(f"Device {i}: {device_name}")

            current_device = torch.cuda.current_device()
            single_print(f"Current device index: {current_device}")
        else:
            num_devices = 0
            single_print("CUDA is not available.")

        for index_nmodel in range(inputs.nmodel):
            for index_nstep in range(inputs.nstep):
                dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
                if os.path.exists(f'{workpath}/{dply_model}'):
                    single_print(f'\t\tFound the deployed model: {dply_model}')
                    if num_devices > 0:
                        index_totalmodel = index_nmodel*int(inputs.nstep) + index_nstep
                        device_index = index_totalmodel%num_devices
                        device_name = f"{inputs.device}:{device_index}"
                        single_print(f"\t\tLoad to device: {device_name}")
                        inputs.calc_MLIP.append(
                                nequip_calculator.NequIPCalculator.from_deployed_model(f'{workpath}/{dply_model}', device=device_name)
                        )
                    else:
                        inputs.calc_MLIP.append(
                            nequip_calculator.NequIPCalculator.from_deployed_model(f'{workpath}/{dply_model}', device=inputs.device)
                        )
                else:
                    # If there is no model, turn on the termination signal
                    single_print(f'\t\tCannot find the model: {dply_model}')
                    signal = 1

        # ensemble_calculate shares one neighbor list across the ensemble,
        # which is only valid if every model has the same r_max and the
        # same TypeMapper transform.
        if signal == 0 and len(inputs.calc_MLIP) > 1:
            ref = inputs.calc_MLIP[0]
            for c in inputs.calc_MLIP[1:]:
                assert c.r_max == ref.r_max, \
                    'Ensemble models have different r_max; ensemble_calculate cannot share neighbor list'
                assert type(c.transform) is type(ref.transform), \
                    'Ensemble models have different TypeMapper types'

    elif inputs.MLIP == 'so3krates':
        from glp import instantiate
        from glp.ase import Calculator
        for index_nmodel in range(inputs.nmodel):
            for index_nstep in range(inputs.nstep):
                potential_dict = {"mlff": {"folder": f"{workpath}/deployed-model_{index_nmodel}_{index_nstep}"}}
                get_calculator = instantiate.get_calculator(potential_dict, {"atom_pair": {"skin": inputs.skin}})
                inputs.calc_MLIP.append(Calculator(get_calculator))

    # from ase.io.aims import read_aims
    # ref = read_aims('/u/kkang/scratch/r_ZrO_ALmoMD_so3krates/4_ALmoMD/geometry.in.supercell')

    # for calculator in inputs.calc_MLIP:
    #     ref.calc = calculator
    #     print(ref.get_potential_energy())

    # Check the termination signal
    if signal == 1:
        single_print('[Termi]\tSome training processes are not finished.')
        sys.exit()

    return inputs
