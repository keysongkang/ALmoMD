import ase.units as units

import os
import pandas as pd

from libs.lib_util import mpi_print
from libs.lib_nvtlangevin import NVTLangevin


def runMD(
    struc, ensemble, temperature, pressure, timestep, friction,
    compressibility, taut, taup, mask, loginterval, steps,
    nstep, nmodel, E_ref, al_type, logfile, trajectory,
    calculator, signal_uncert, signal_append
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
            nstep = nstep,
            nmodel = nmodel,
            calculator = calculator,
            E_ref = E_ref,
            al_type = al_type,
            trajectory = trajectory,
            logfile = logfile,
            signal_uncert = signal_uncert,
            signal_append = signal_append
        )
    else:
        from mpi4py import MPI
        # Extract MPI infos
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        
        mpi_print(f'The ensemble model is not determined.', rank)
