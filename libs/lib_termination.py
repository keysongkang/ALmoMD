import os
import numpy as np
import pandas as pd
from libs.lib_util        import single_print


def termination(temperature, pressure, crtria_cnvg, NumAtoms):
    """Function [termination]
    Activate the termination signal upon satisfying the criteria

    Parameters:

    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    crtria: float
        Convergence criteria
    crtria_cnvg: float
        Convergence criteria ##!! We need to replace crtria_cnvg to crtria
    NumAtoms: int
        The number of atoms in the simulation cell

    Returns:

    signal: int
        The termination signal
    """

    # Initialization
    signal = 0

    # Read the result file
    if os.path.exists('result.txt'):
        result_data = pd.read_csv(
            'result.txt', index_col=False, delimiter='\t'
            )
        # Read the Uncertainty column
        result_uncert = result_data.loc[:,'UncertAbs_All']

        # Verify three consecutive results against the convergence criteria
        if len(result_uncert) > 2:
            result_max = max(result_uncert[-3:])
            result_min = min(result_uncert[-3:])

            # Currently criteria is written by eV/atom
            if np.absolute(result_max - result_min) < crtria_cnvg * NumAtoms and \
            result_max != result_min:
                single_print(
                    f'Converged result at temperature {temperature}K\n:'
                    +f'{np.absolute(result_max-result_min)}'
                    )
                signal = 1
            else:
                signal = 0
        
    return signal
