import ase.units as units

import os
import pandas as pd

from mpi4py import MPI
from libs.lib_util import mpi_print
from libs.lib_nvtlangevin import NVTLangevin


def runMD(
    struc, ensemble, temperature, pressure, timestep, friction,
    compressibility, taut, taup, mask, loginterval, steps,
    nstep, nmodel, logfile, trajectory, calculator
):
    """Function [runMD]
    Initiate the Molecular Dynamics simulation using various ensembles

    Parameters:

    struc: ASE atoms
        A structral configuration of a starting point
    ensemble: str
        Type of MD ensembles; 'NVTLangevin'
    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    timestep: float
        The step interval for printing MD steps

    friction: float
        Strength of the friction parameter in NVTLangevin ensemble
    compressibility: float
        compressibility in units of eV/Angstrom**3 in NPTBerendsen
    taut: float
        Time constant for Berendsen temperature coupling
        in NVTBerendsen and NPT Berendsen
    taup: float
        Time constant for Berendsen pressure coupling in NPTBerendsen
    mask: Three-element tuple
        Dynamic elements of the computational box (x,y,z);
        0 is false, 1 is true

    loginterval: int
        The step interval for printing MD steps
    steps: int
        The length of the Molecular Dynamics steps
    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization

    logfile: str
        A name of MD logfile
    trajectory: str
        A name of MD trajectory file
    calculator: ASE calculator
        Calculators from trained models
    """

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
    nmodel, index, calculator
):
    """Function [check_runMD]
    After confirming the existence of a previously completed MD calculation,
    initiate the Molecular Dynamics simulation using various ensembles.

    Parameters:

    struc: ASE atoms
        A structral configuration of a starting point
    ensemble: str
        Type of MD ensembles; 'NVTLangevin'
    temperature: float
        The desired temperature in units of Kelvin (K)
    pressure: float
        The desired pressure in units of eV/Angstrom**3
    timestep: float
        The step interval for printing MD steps

    friction: float
        Strength of the friction parameter in NVTLangevin ensemble
    compressibility: float
        compressibility in units of eV/Angstrom**3 in NPTBerendsen
    taut: float
        Time constant for Berendsen temperature coupling
        in NVTBerendsen and NPT Berendsen
    taup: float
        Time constant for Berendsen pressure coupling in NPTBerendsen
    mask: Three-element tuple
        Dynamic elements of the computational box (x,y,z);
        0 is false, 1 is true

    loginterval: int
        The step interval for printing MD steps
    steps: int
        The length of the Molecular Dynamics steps
    nstep: int
        The number of subsampling sets
    nmodel: int
        The number of ensemble model sets with different initialization

    logfile: str
        A name of MD logfile
    trajectory: str
        A name of MD trajectory file
    calculator: ASE calculator
        Calculators from trained models

    Returns:

    trajectory: str
        A name of MD trajectory file at the current step
    """

    # Extract MPI infos
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Prepare the nmae of MD trajectory and log files
    trajectory = f'traj-{temperature}K-{pressure}bar_{index}.traj'
    logfile    = f'traj-{temperature}K-{pressure}bar_{index}.log'
    
    mpi_print(f'Checking the MD_{index} outputs...\n', rank)
    
    if os.path.exists(trajectory): # When there is the previous trajectory file
        mpi_print(f'Found the MD_{index} outputs: {trajectory}\n', rank)
        # Open the MD log file
        MD_data = pd.read_csv(logfile, index_col=False, delim_whitespace=True)
        MD_length = len(MD_data.loc[:,'Time[ps]']); del MD_data
        
        if MD_length != int(steps/loginterval)+1: # If MD is not finished,
            mpi_print(f'It seems this MD is terminated in middle.\n', rank)
            if rank == 0:
                # Remove pervious output
                os.system(f'rm {trajectory} {logfile}')
            mpi_print(
                f'Initiate a Molecular Dynamics calculation for'
                + f'{trajectory}...\n', rank
            )
            # Perform the MD calculations again from the beginning
            runMD(
                struc, ensemble, temperature, pressure, timestep, friction,
                compressibility, taut, taup, mask, loginterval, steps, nstep,
                nmodel, logfile, trajectory, calculator
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
        # Implement the MD calculation
        runMD(
            struc, ensemble, temperature, pressure, timestep, friction,
            compressibility, taut, taup, mask, loginterval, steps, nstep,
            nmodel, logfile, trajectory, calculator
        )
        mpi_print(
            f'Finish the Molecular Dynamics calculation for'
            + f'{trajectory}...\n', rank
        )
        
    return trajectory