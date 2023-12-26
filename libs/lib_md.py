import ase.units as units

import os
import pandas as pd

from libs.lib_util import mpi_print
from libs.lib_nvtlangevin import NVTLangevin
from libs.lib_nvtlangevin_meta import NVTLangevin_meta
from libs.lib_nptisoiso import NPTisoiso

def runMD(
    inputs, struc, steps,
    logfile, trajectory, calculator,
    signal_uncert, signal_append
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

    if inputs.ensemble == 'NVTLangevin':
        NVTLangevin(
            struc = struc,
            timestep = inputs.timestep * units.fs,
            temperature = inputs.temperature * units.kB,
            friction = inputs.friction,
            steps = steps,
            loginterval = inputs.loginterval,
            nstep = inputs.nstep,
            nmodel = inputs.nmodel,
            calculator = calculator,
            E_ref = 0.0,
            al_type = inputs.al_type,
            trajectory = trajectory,
            harmonic_F = inputs.harmonic_F,
            anharmonic_F = inputs.anharmonic_F,
            logfile = logfile,
            signal_uncert = signal_uncert,
            signal_append = signal_append
        )
    elif inputs.ensemble == 'NVTLangevin_meta':
        NVTLangevin_meta(
            struc = struc,
            timestep = inputs.timestep * units.fs,
            temperature = inputs.temperature * units.kB,
            friction = inputs.friction,
            steps = steps,
            loginterval = inputs.loginterval,
            nstep = inputs.nstep,
            nmodel = inputs.nmodel,
            calculator = calculator,
            E_ref = 0.0,
            al_type = inputs.al_type,
            trajectory = trajectory,
            meta_Ediff = inputs.meta_Ediff,
            harmonic_F = inputs.harmonic_F,
            anharmonic_F = inputs.anharmonic_F,
            logfile = logfile,
            signal_uncert = signal_uncert,
            signal_append = signal_append
        )
    elif inputs.ensemble == 'NPTisoiso':
        NPTisoiso(
            struc = struc,
            timestep = inputs.timestep * units.fs,
            temperature = inputs.temperature * units.kB,
            pressure = inputs.pressure * units.GPa,
            ttime = inputs.ttime * units.fs,
            pfactor = inputs.pfactor,
            mask = inputs.mask,
            steps = steps,
            loginterval = inputs.loginterval,
            nstep = inputs.nstep,
            nmodel = inputs.nmodel,
            calculator = calculator,
            E_ref = 0.0,
            al_type = inputs.al_type,
            trajectory = trajectory,
            harmonic_F = inputs.harmonic_F,
            anharmonic_F = inputs.anharmonic_F,
            logfile = logfile,
            signal_uncert = signal_uncert,
            signal_append = signal_append
        )
    else:        
        mpi_print(f'The ensemble model is not determined.', inputs.rank)
