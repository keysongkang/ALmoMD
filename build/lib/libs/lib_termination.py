import os
import numpy as np
import pandas as pd
from libs.lib_util        import single_print


def termination(temperature, pressure, crtria_cnvg, NumAtoms):
    signal = 0
    if os.path.exists('result.txt'):
        result_data = pd.read_csv('result.txt', index_col=False,\
                                  delimiter='\t')
        result_uncert = result_data.loc[:,'UncertAbs_All']

        if len(result_uncert) > 2:
            result_max = max(result_uncert[-3:])
            result_min = min(result_uncert[-3:])

            if np.absolute(result_max-result_min) < crtria_cnvg*NumAtoms and\
            result_max !=result_min:
                single_print(
                    f'Converged result at temperature {temperature}K\n:'
                    +f'{np.absolute(result_max-result_min)}'
                    )
                signal = 1
            else:
                signal = 0
        
    return signal
