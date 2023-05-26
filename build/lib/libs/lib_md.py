import ase.units as units

import os
import pandas as pd

from libs.lib_util import mpi_print
from libs.lib_nvtlangevin import NVTLangevin


def runMD(
    struc, ensemble, temperature, pressure, timestep, friction,
    compressibility, taut, taup, mask, loginterval, steps, nstep, nmodel,
    logfile, trajectory, calculator, comm, size, rank
):
    if ensemble == 'NVTLangevin':
        NVTLangevin(
            struc = struc,
            timestep = timestep * units.fs,
            temperature = temperature * units.kB,
            friction = friction,
            steps = steps,
            loginterval = loginterval,
            logfile = logfile,
            trajectory = trajectory,
            nstep = nstep,
            nmodel = nmodel,
            calculator = calculator,
            comm = comm,
            size = size,
            rank = rank
        )
    else:
        mpi_print(f'The ensemble model is not determined.', rank)


def check_runMD(
    struc, ensemble, temperature, pressure, timestep, friction,
    compressibility, taut, taup, mask, loginterval, steps, nstep,
    nmodel, index, calculator, comm, size, rank
):
    trajectory = f'traj-{temperature}K-{pressure}bar_{index}.traj'
    logfile    = f'traj-{temperature}K-{pressure}bar_{index}.log'
    workpath   = f'./data/{temperature}K-{pressure}bar_{index}'
    
    mpi_print(f'Checking the MD_{index} outputs...\n', rank)
    
    if os.path.exists(trajectory):
        mpi_print(f'Found the MD_{index} outputs: {trajectory}\n', rank)
        MD_data = pd.read_csv(logfile, index_col=False, delim_whitespace=True)
        MD_length = len(MD_data.loc[:,'Time[ps]']); del MD_data
        
        if MD_length != int(steps/loginterval)+1:
            mpi_print(f'It seems this MD is terminated in middle.\n', rank)
            if rank == 0:
                os.system(f'rm {trajectory} {logfile}')
            mpi_print(
                f'Initiate a Molecular Dynamics calculation for'
                + f'{trajectory}...\n', rank
            )
            runMD(
                struc, ensemble, temperature, pressure, timestep, friction,\
                compressibility, taut, taup, mask, loginterval, steps, nstep,\
                nmodel, logfile, trajectory, calculator, comm, size, rank
            )
            mpi_print(
                f'Finish the Molecular Dynamics calculation for'
                + f'{trajectory}...\n', rank
            )
    else:
        mpi_print(
            f'Initiate a Molecular Dynamics calculation for'
            + f'{trajectory}...\n', rank
        )
        runMD(
            struc, ensemble, temperature, pressure, timestep, friction,\
            compressibility, taut, taup, mask, loginterval, steps, nstep,\
            nmodel, logfile, trajectory, calculator, comm, size, rank
        )
        mpi_print(
            f'Finish the Molecular Dynamics calculation for'
            + f'{trajectory}...\n', rank
        )
        
    return trajectory