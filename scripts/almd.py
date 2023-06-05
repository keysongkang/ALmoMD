import os
import son
import sys

import numpy as np
from mpi4py import MPI
from decimal import Decimal
from nequip.ase import nequip_calculator

from ase import Atoms
from ase.build import make_supercell
from ase.io import read as atoms_read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from libs.lib_util import check_mkdir, job_dependency, read_input_file, mpi_print, single_print, output_init
from libs.lib_npz import generate_npz_DFT_init, generate_npz_DFT, generate_npz_DFT_rand_init, generate_npz_DFT_rand
from libs.lib_train import get_train_job, execute_train_job
from libs.lib_dft import run_DFT
from libs.lib_progress    import check_progress, check_index
from libs.lib_mainloop    import MLMD_initial, MLMD_main, MLMD_random
from libs.lib_criteria    import get_result, get_criteria
from libs.lib_termination import termination

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import r2_score, mean_absolute_error

import torch
torch.set_default_dtype(torch.float64)


class almd:
    """Object [almd]
    Main script of ALmoMD.

    Functions:

    __init__: Automatically initialize the inputs and read input.in file.
    run_dft_init: Perform the calculation by initializing with the aiMD trajectory.
    run_dft_cont: Run and continue the ALMD calculation.
    run_dft_gen: Extract DFT results, generate the training data, and execute NequIP.
    run_dft_rand: Implement the ALMD calculation using random samplings.
    run_dft_test: Check the validation error.
    """

    def __init__(self, input_file='input.in'):
        """Function [__init__]
        Automatically initialize the inputs and read input.in file.

        Parameters:

        input_file: Read the inputs from input.in file 
        """

        # Extract MPI infos
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        ### Version
        self.version = '0.0.0'

        ### Default setting for inputs
        ##[Active learning types]
        # calc_type: str
        #     Type of sampling; 'active' (active learning), 'random', 'converge'
        self.calc_type = 'active'
        # al_type: str
        #     Type of active learning; 'energy', 'force', 'force_max'
        self.al_type = 'force'
        # uncert_type: str
        #     Type of uncertainty; 'absolute', 'relative'
        self.uncert_type = 'absolute'
        # output_format: str
        #     Type of FHI-vibes output to be read
        self.output_format = 'trajectory.son'

        ##[Active learning setting]
        # nstep: int
        #     The number of subsampling sets
        self.nstep = 1
        # nmodel: int
        #     The number of ensemble model sets with different initialization
        self.nmodel = 1
        # steps_init: int
        #     Initialize MD steps to get averaged uncertainties and energies
        self.steps_init = 125
        # steps_random: int
        #     the length of MD run for random sampling
        self.steps_random = 125
        # ntrain_init: int
        #     The number of training data for first iterative step
        self.ntrain_init = 5
        # ntrain: int
        #     The number of added training data for each iterative step
        self.ntrain = 5
        # crtria: float
        #     Convergence criteria
        self.crtria = 0.0000001
        self.crtria_cnvg = 0.0000001 ##!! This is needed to be removed.
        # num_calc: int
        #     The number of job scripts for DFT calculations
        self.num_calc = 20

        ##[Molecular dynamics setting]
        # ensemble: str
        #     Type of MD ensembles; 'NVTLangevin'
        self.ensemble = 'NVTLangevin'
        # supercell: 2D array of floats
        #     Tensor shape of supercell for MD calculations
        #     compared to primitive cell.
        self.supercell = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        # timestep: float
        #     MD timestep in units of femto-second (fs)
        self.timestep = 1
        # loginterval: int
        #     The step interval for printing MD steps
        self.loginterval = 10
        # temperature: float
        #     The desired temperature in units of Kelvin (K)
        self.temperature = 2000
        # pressure: float
        #     The desired pressure in units of eV/Angstrom**3
        self.pressure = 0

        ##[runMD default inputs]
        # workpath: str
        #     A path to the directory containing trained models
        self.modelpath = './MODEL'
        # logfile: str
        #     A file name for MD logging
        self.logfile = 'md.log'
        # trajectory: str
        #     A file name for MD trajectory
        self.trajectory = 'md.traj'
        # steps: int
        #     A target simulation timesteps
        self.steps = 500
        # MD_input: str
        #     A file name of the input file for initial structure
        #     'geometry.in', 'geometry.in.next_step', 'trajectory.son' work
        self.MD_input = 'trajectory.son'
        # supercell_init: 2D array of floats
        #     Tensor shape of supercell for runMD calculations
        #     compared to primitive cell.
        self.supercell_init = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        ##[NVTLangevin setting]
        # friction: float
        #     Strength of the friction parameter in NVTLangevin ensemble
        self.friction = 0.02

        ##[NPTBerendsen setting]
        # taut: float
        #     Time constant for Berendsen temperature coupling
        #     in NVTBerendsen and NPT Berendsen
        self.taut = 50
        # compressibility: float
        #     compressibility in units of eV/Angstrom**3 in NPTBerendsen
        self.compressibility = 4.57e-5
        # taup: float
        #     Time constant for Berendsen pressure coupling in NPTBerendsen
        self.taup = 100
        # mask: Three-element tuple
        #     Dynamic elements of the computational box (x,y,z); 0 is false, 1 is true
        self.mask = (1, 1, 1)

        ##[NequIP setting]
        # rmax: float
        #     Cutoff radius in NequIP
        self.rmax = 3.5
        # lmax: int
        #     Maximum angular momentum in NequIP
        self.lmax = 2
        # nfeatures: int
        #     The number of features used in NequIP
        self.nfeatures = 16
        # E_gs: float
        #     Reference total energy in units of eV/Unitcell
        #     to shift the total energies of the trajectory
        #     and avoid unusually high total energy values
        #     that may lead to unusual weightings with force values.
        #     Recommend to use the ground state total energy.
        self.E_gs = 0.0

        ##[Constants]
        # kB: float
        #     Boltzmann constant in units of eV/K
        self.kB = 8.617333262e-5


        ### Implement reading the input file
        # Read input variables from the input file
        input_variables = read_input_file(input_file)
        
        # Update instance attributes with the input variables
        self.__dict__.update(input_variables)
        

        ### Perform postprocessing of inputs
        # Validation steps are currently fixed for 16.67% (1/6) of total data.
        # nval_init: The number of validation data for first iterative step
        self.nval_init = int(self.ntrain_init / 5)
        # nval: The number of added validation data for each iterative step
        self.nval = int(self.ntrain / 5)
        # ntotal_init: Total number of training and valdiation data for all subsamplings for first iteractive step
        self.ntotal_init = (self.ntrain_init + self.nval_init) * self.nstep
        # ntotal: Total number of added training and valdiation data for all subsamplings for each iteractive step
        self.ntotal = (self.ntrain + self.nval) * self.nstep

        ### Initization of parameters
        # index: int
        #     The index of AL interactive steps
        self.index = 0
    


    def run_dft_init(self):
        """Function [run_dft_init]
        Perform the calculation by initializing with the aiMD trajectory.
        """

        # Extract MPI infos
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Print the head
        output_init('init', self.version, MPI=True)
        mpi_print(f'[init]\tInitiate the active learning process', rank)

        # Read aiMD trajectory file of training data
        metadata, traj = son.load('trajectory_train.son')
        mpi_print(f'[init]\tRead the initial trajectory data: trajectory_train.son', rank)

        # Set the path to folders storing the training data for NequIP
        workpath = f'./MODEL/{self.temperature}K-{self.pressure}bar_{self.index}'
        # Create these folders
        if rank == 0:
            check_mkdir('MODEL')
            check_mkdir(workpath)
        comm.Barrier()

        # As it is an initialization step,
        # the total number of training and validation data matches the initial settings
        total_ntrain = self.ntrain_init
        total_nval = self.nval_init

        # Generate first set of training data in npz files from trajectory file
        if rank == 0:
            generate_npz_DFT_init(
                traj, self.ntrain_init, self.nval_init,
                self.nstep, self.E_gs, workpath
            )
        del traj  # Remove it to reduce the memory usage
        mpi_print(f'[init]\tGenerate the training data (# of data: {total_ntrain+total_nval})', rank)
        comm.Barrier()

        # Training process: Run NequIP
        execute_train_job(
            total_ntrain, total_nval, self.rmax, self.lmax, self.nfeatures,
            workpath, self.nstep, self.nmodel
        )
        mpi_print(f'[init]\tSubmit the NequIP training processes', rank)
        comm.Barrier()

        # Submit a job-dependence to execute run_dft_cont after the NequIP training
        # For 'converge' setting, we don't needo to submit it.
        if not self.calc_type == 'converge':
            if rank == 0:
                job_dependency('cont', self.nmodel)

        comm.Barrier()
        mpi_print(f'[init]\t!! Finish the initialization', rank)
    


    def run_dft_cont(self):
        """Function [run_dft_cont]
        Run and continue the ALMD calculation.
        """
        from datetime import datetime

        # Extract MPI infos
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        mpi_print(f'[cont]\t' + datetime.now().strftime("Date/Time: %Y %m %d %H:%M"), rank)
        mpi_print(f'[cont]\tALmoMD Version: {self.version}', rank)
        mpi_print(f'[cont]\tContinue from the previous step (Mode: {self.calc_type})', rank)
        comm.Barrier()

        ## Prepare the ground state structure
        # Read the ground state structure with the primitive cell
        struc_init = atoms_read('geometry.in.next_step', format='aims')
        # Make it supercell
        struc_relaxed = make_supercell(struc_init, self.supercell)
        # Get the number of atoms in the simulation cell
        self.NumAtoms = len(struc_relaxed)
        mpi_print(f'[cont]\tRead the reference structure: geometry.in.next_step', rank)
        comm.Barrier()

        ### Initizlization step
        ##!! We need to check whether this one is needed or not.
        # Get total_index to resume the MLMD calculation
        if rank == 0:
            # Open the result.txt file
            self.index = check_index(self.index)
        self.index = comm.bcast(self.index, root=0)
        comm.Barrier()

        ## For active learning sampling,
        if self.calc_type == 'active':
            # Retrieve the calculation index (kndex: MLMD_initial, MD_index: MLMD_main, signal: termination)
            # to resume the MLMD calculation if a previous one exists.
            kndex, MD_index, signal = None, None, None

            # Open the uncertainty output file
            kndex, MD_index, self.index, signal = check_progress(
                self.temperature, self.pressure,
                self.ntotal, self.ntrain, self.nval,
                self.nstep, self.nmodel, self.steps_init,
                self.index, self.crtria, self.NumAtoms
            )
            kndex = comm.bcast(kndex, root=0)
            MD_index = comm.bcast(MD_index, root=0)
            self.index = comm.bcast(self.index, root=0)
            self.temperature = comm.bcast(self.temperature, root=0)
            signal = comm.bcast(signal, root=0)
        ## For random sampling,
        elif self.calc_type == 'random':
            # Here, just resume the MLMD for a specific index if a previous one exists
            kndex = 0
            MD_index = 0
            signal = 0   ##!! Termination check should be added here
            if rank == 0:
                self.index = check_index(self.index)
            self.index = comm.bcast(self.index, root=0)
        comm.Barrier()

        # If we get the signal from check_progress, the script will be terminated.
        if signal == 1:
            mpi_print(f'[cont]\tCalculation is terminated during the check_progress', rank)
            sys.exit()

        mpi_print(f'[cont]\tCurrent iteration index: {self.index}', rank)
        # Get the total number of traning and validating data at current step
        total_ntrain = self.ntrain * self.index + self.ntrain_init
        total_nval = self.nval * self.index + self.nval_init
        comm.Barrier()


        ### Get calculators
        # Set the path to folders finding the trained model from NequIP
        workpath = f'./MODEL/{self.temperature}K-{self.pressure}bar_{self.index}'
        # Create these folders
        if rank == 0:
            check_mkdir('MODEL')
            check_mkdir(workpath)
        comm.Barrier()

        mpi_print(f'[cont]\tFind the trained models: {workpath}', rank)
        # Get calculators from previously trained MLIP and its total energy of ground state structure
        E_ref, calc_MLIP = get_train_job(
            struc_relaxed, total_ntrain, total_nval, self.rmax, self.lmax,
            self.nfeatures, workpath, self.nstep, self.nmodel
        )
        comm.Barrier()

        mpi_print(f'[cont]\tImplement MD for active learning (Mode: {self.calc_type})', rank)
        ### MLMD steps
        # For active learning sampling,
        if self.calc_type == 'active':
            # Run MLMD calculation to extract avaerged uncertainty and potential energy
            ##!! struc_step might be able to be removed.
            struc_step = MLMD_initial(
                kndex, self.index, self.ensemble, self.temperature, self.pressure, self.timestep, self.friction,
                self.compressibility, self.taut, self.taup, self.mask, self.loginterval, self.steps_init, self.nstep,
                self.nmodel, calc_MLIP, E_ref, self.al_type
            )
            # Run MLMD calculation with active learning sampling
            MLMD_main(
                MD_index, self.index, self.ensemble, self.temperature, self.pressure, self.timestep, self.friction,
                self.compressibility, self.taut, self.taup, self.mask, self.loginterval, self.ntotal, self.nstep,
                self.nmodel, calc_MLIP, E_ref, self.steps_init, self.NumAtoms, self.kB, struc_step,
                self.al_type, self.uncert_type
            )
        # For random sampling,
        elif self.calc_type == 'random':
            # Run MLMD calculation with random sampling
            struc_step = MLMD_random(
                kndex, self.index, self.ensemble, self.temperature, self.pressure, self.timestep, self.friction,
                self.compressibility, self.taut, self.taup, self.mask, self.loginterval, self.steps_random,
                self.nstep, self.nmodel, calc_MLIP, E_ref
            )
        else:
            raise ValueError("[cont]\tInvalid calc_type. Supported values are 'active' and 'random'.")
        comm.Barrier()

        mpi_print(f'[cont]\tRecord the results: result.txt', rank)
        # Record uncertainty results at the current step
        get_result(self.temperature, self.pressure, self.index, self.steps_init)
        comm.Barrier()

        mpi_print(f'[cont]\tSubmit the DFT calculations for sampled configurations', rank)
        # Submit job-scripts for DFT calculations with sampled configurations and job-dependence for run_dft_gen
        if rank == 0:
            run_DFT(self.temperature, self.pressure, self.index, (self.ntrain + self.nval) * self.nstep, self.num_calc)
        comm.Barrier()

        # Submit a job-dependence to execute run_dft_gen after the DFT calculations
        if rank == 0:
            job_dependency('gen', self.num_calc)
        comm.Barrier()

        mpi_print(f'[cont]\t!! Finish the MD investigation: Iteration {index}', rank)


    def run_dft_gen(self):
        """Function [run_dft_gen]
        Extract DFT results, generate the training data, and execute NequIP.
        """
        from datetime import datetime

        # Extract MPI infos
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        mpi_print(f'[gen]\t' + datetime.now().strftime("Date/Time: %Y %m %d %H:%M"), rank)
        mpi_print(f'[gen]\tALmoMD Version: {self.version}', rank)
        mpi_print(f'[gen]\tGenerate the NequIP inputs from DFT results', rank)
        comm.Barrier()

        ## Prepare the ground state structure
        # Read the ground state structure with the primitive cell
        struc_init = atoms_read('geometry.in.next_step', format='aims')
        # Make it supercell
        struc_relaxed = make_supercell(struc_init, self.supercell)
        # Get the number of atoms in unitcell
        self.NumAtoms = len(struc_relaxed)
        mpi_print(f'[cont]\tRead the reference structure: geometry.in.next_step', rank)
        comm.Barrier()

        ### Initizlization step
        ##!! We need to check whether this one is needed or not.
        # Get total_index to resume the MLMD calculation
        if rank == 0:
            self.index = check_index(self.index)
        self.index = comm.bcast(self.index, root=0)

        ## For active learning sampling,
        if self.calc_type == 'active':
            # Retrieve the calculation index (kndex: MLMD_initial, MD_index: MLMD_main, signal: termination)
            # to resume the MLMD calculation if a previous one exists.
            kndex, MD_index, signal = None, None, None
            # Open the uncertainty output file
            kndex, MD_index, self.index, signal = check_progress(
                self.temperature, self.pressure,
                self.ntotal, self.ntrain, self.nval,
                self.nstep, self.nmodel, self.steps_init,
                self.index, self.crtria, self.NumAtoms
            )
            kndex = comm.bcast(kndex, root=0)
            MD_index = comm.bcast(MD_index, root=0)
            self.index = comm.bcast(self.index, root=0)
            self.temperature = comm.bcast(self.temperature, root=0)
            signal = comm.bcast(signal, root=0)
        ## For random sampling,
        elif self.calc_type == 'random':
            # Here, just resume the MLMD for a specific index if a previous one exists
            kndex = 0
            MD_index = 0
            signal = 0
            total_index = None
            if rank == 0:
                self.index = check_index(self.index)
            self.index = comm.bcast(self.index, root=0)
        else:
            single_print('You need to assign calc_type.')

        # If we get the signal from check_progress, the script will be terminated.
        if signal == 1:
            mpi_print(f'[gen]\tCalculation is terminated during the check_progress', rank)
            sys.exit()
        comm.Barrier()

        mpi_print(f'[gen]\tCurrent iteration index: {self.index}', rank)
        # Get the total number of traning and validating data at current step
        total_ntrain = self.ntrain * self.index + self.ntrain_init
        total_nval = self.nval * self.index + self.nval_init


        ### Get DFT results and generate training data
        # Set the path to folders storing the training data for NequIP
        workpath = f'./MODEL/{self.temperature}K-{self.pressure}bar_{self.index}'
        if rank == 0:
            check_mkdir('MODEL')
            check_mkdir(workpath)
        comm.Barrier()

        mpi_print(f'[gen]\tGenerate the training data from DFT results (# of data: {total_ntrain+total_nval})', rank)
        # Generate first set of training data in npz files from trajectory file
        if rank == 0:
            generate_npz_DFT(
                self.ntrain, self.nval, self.nstep, self.E_gs, self.index, self.temperature,
                self.output_format, self.pressure, workpath
            )
        comm.Barrier()

        # Training process: Run NequIP
        execute_train_job(
            total_ntrain, total_nval, self.rmax, self.lmax, self.nfeatures,
            workpath, self.nstep, self.nmodel
        )
        comm.Barrier()
        mpi_print(f'[gen]\tSubmit the NequIP training processes', rank)

        # Submit a job-dependence to execute run_dft_cont after the NequIP training
        if rank == 0:
            job_dependency('cont', self.nmodel)
        comm.Barrier()

        mpi_print(f'[gen]\t!! Finish the training data generation: Iteration {index}', rank)



    def run_dft_rand(self):
        """Function [run_dft_rand]
        Implement the ALMD calculation using random samplings.
        """

        # Extract MPI infos
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Print the head
        output_init('rand', self.version, MPI=True)
        mpi_print(f'[rand]\tInitiate the random sampling process', rank)

        # Read aiMD trajectory file of training data
        metadata, traj = son.load('trajectory_train.son')
        mpi_print(f'[rand]\tRead the initial trajectory data: trajectory_train.son', rank)
        comm.Barrier()

        # As it is an initialization step,
        # the total number of training and validation data matches the initial settings
        total_ntrain = self.ntrain_init
        total_nval = self.nval_init

        # Start from the first step
        index = 0
        # Set the path to folders storing the training data for NequIP
        workpath = f'./MODEL/{self.temperature}K-{self.pressure}bar_{index}'

        if rank == 0:
            # Create these folders
            check_mkdir('MODEL')
            check_mkdir(workpath)

            # Generate first set of training data in npz files from trajectory file
            traj_idx = generate_npz_DFT_rand_init(
                traj, self.ntrain_init, self.nval_init,
                self.nstep, self.E_gs, workpath
            )
        mpi_print(f'[rand]\tGenerate the training data from trajectory_train.son (# of data: {total_ntrain+total_nval})', rank)
        comm.Barrier()

        # Training process: Run NequIP
        execute_train_job(
            total_ntrain, total_nval, self.rmax, self.lmax, self.nfeatures,
            workpath, self.nstep, self.nmodel
        )
        mpi_print(f'[rand]\tSubmit the NequIP training processes for iteration {index}', rank)
        comm.Barrier()

        # Run steps until random_index (which is assigned in input.in)
        while index < self.random_index:
            # Go to the next step
            index += 1

            # Add the number of new training and validating data
            total_ntrain += self.ntrain
            total_nval += self.nval

            # Set the path to folders storing the training data for NequIP at the current step
            workpath = f'./MODEL/{self.temperature}K-{self.pressure}bar_{index}'

            if rank == 0:
                # Create these folders
                check_mkdir(workpath)

                # Generate first set of training data in npz files from trajectory file
                traj_idx = generate_npz_DFT_rand(
                    traj, self.ntrain, self.nval, self.nstep, self.E_gs, index, self.temperature,
                    self.pressure, workpath, traj_idx
                )
            comm.Barrier()
            mpi_print(f'[rand]\tGenerate the training data from trajectory_train.son (# of data: {total_ntrain+total_nval})', rank)

            # Training process: Run NequIP
            execute_train_job(
                total_ntrain, total_nval, self.rmax, self.lmax, self.nfeatures,
                workpath, self.nstep, self.nmodel
            )
            comm.Barrier()
            mpi_print(f'[rand]\tSubmit the NequIP training processes for iteration {index}', rank)

        mpi_print(f'[rand]\t!! Finish the random sampling process', rank)


    def run_dft_test(self):
        """Function [run_dft_test]
        Check the validation error.
        """

        # Extract MPI infos
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # Print the head
        output_init('test', self.version, MPI=True)
        mpi_print(f'[test]\tInitiate the validation test process', rank)
        comm.Barrier()

        # Read testing data
        npz_test = f'./MODEL/data-test.npz' # Path
        data_test = np.load(npz_test)         # Load
        NumAtom = len(data_test['z'][0])      # Get the number of atoms in the simulation cell
        mpi_print(f'[test]\tRead the testing data: data-test.npz', rank)
        comm.Barrier()

        # Set the path to folders storing the training data for NequIP
        workpath = f'./MODEL/{self.temperature}K-{self.pressure}bar_{self.test_index}'
        # Initialization of a termination signal
        signal = 0

        mpi_print(f'[test]\tFind the trained models: {workpath}', rank)
        # Load the trained models as calculators
        calc_MLIP = []
        for index_nmodel in range(self.nmodel):
            for index_nstep in range(self.nstep):
                if (index_nmodel * self.nstep + index_nstep) % size == rank:
                    dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
                    if os.path.exists(f'{workpath}/{dply_model}'):
                        mpi_print(f'\t\tFound the deployed model: {dply_model}', rank)
                        calc_MLIP.append(
                            nequip_calculator.NequIPCalculator.from_deployed_model(f'{workpath}/{dply_model}')
                        )
                    else:
                        # If there is no model, turn on the termination signal
                        mpi_print(f'\t\tCannot find the model: {dply_model}', rank)
                        signal = 1
                        signal = comm.bcast(signal, root=rank)

        # Check the termination signal
        if signal == 1:
            mpi_print('[test]\tSome training processes are not finished.', rank)
            sys.exit()
        comm.Barrier()

        # Open the file to store the results
        if rank == 0:
            outputfile = open(f'result-test_{self.test_index}_energy.txt', 'w')
            outputfile.write('index   \tUncertAbs\tRealErrorAbs\tRealE\tPredictE\n')
            outputfile.close()

        comm.Barrier()
        mpi_print(f'[test]\tGo through all configurations in the testing data ...', rank)
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
            for index_nmodel in range(self.nmodel):
                for index_nstep in range(self.nstep):
                    if (index_nmodel * self.nstep + index_nstep) % size == rank:
                        struc.calc = calc_MLIP[zndex]
                        prd_E.append(struc.get_potential_energy())
                        prd_F.append(struc.get_forces())
                        zndex += 1

            # Get average and standard deviation (Uncertainty) of predicted energies from various models
            prd_E = comm.allgather(prd_E)
            prd_E_avg = np.average([jtem for item in prd_E if len(item) != 0 for jtem in item], axis=0)
            prd_E_std = np.std([jtem for item in prd_E if len(item) != 0 for jtem in item], axis=0)

            # Get the real error
            realerror_E = np.absolute(prd_E_avg - id_E)

            # Get average of predicted forces from various models
            prd_F = comm.allgather(prd_F)
            prd_F_avg = np.average([jtem for item in prd_F if len(item) != 0 for jtem in item], axis=0)

            # Save all energy information
            if rank == 0:
                trajfile = open(f'result-test_{self.test_index}_energy.txt', 'a')
                trajfile.write(
                    str(config_idx) + '          \t' +
                    '{:.5e}'.format(Decimal(str(prd_E_std))) + '\t' +       # Standard deviation (Uncertainty)
                    '{:.5e}'.format(Decimal(str(realerror_E))) + '\t' +     # Real error
                    '{:.10e}'.format(Decimal(str(id_E))) + '\t' +           # Real energy
                    '{:.10e}'.format(Decimal(str(prd_E_avg))) + '\n'        # Predicted energy
                )
                trajfile.close()

            # Save all force information
            if rank == 0:
                trajfile = open(f'result-test_{self.test_index}_force.txt', 'a')
                for kndex in range(len(prd_F_avg)):
                    for lndex in range(3):
                        trajfile.write(
                            '{:.10e}'.format(Decimal(str(id_F[kndex][lndex]))) + '\t' +     # Real force
                            '{:.10e}'.format(Decimal(str(prd_F_avg[kndex][lndex]))) + '\n'  # Predicted force
                        )
                trajfile.close()

            comm.Barrier()
            config_idx += 1

        mpi_print(f'[test]\tPlot the results ...', rank)
        ## Plot the energy and force prediction results
        # Read the energy data
        data = pd.read_csv(f'result-test_{self.test_index}_energy.txt', sep="\t")

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
        data = pd.read_csv(f'result-test_{self.test_index}_force.txt', sep="\t", header=None)

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
        mpi_print(f'[test]\t[[Statistic results]]', rank)
        mpi_print(f'[test]\tEnergy_R2\tEnergy_MAE\tForces_R2\tForces_MAE', rank)
        mpi_print(f'[test]\t{R2_E}\t{MAE_E}\t{R2_F}\t{MAE_F}', rank)

        mpi_print(f'[test]\t!! Finish the testing process', rank)


    def run_dft_runmd(self):
        """Function [run_dft_runmd]
        Initiate MD calculation using trained models.
        """
        from libs.lib_md import runMD
        from ase.data   import atomic_numbers

        # Extract MPI infos
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # Print the head
        output_init('runMD', self.version, MPI=True)
        mpi_print(f'[runMD]\tInitiate runMD process', rank)
        comm.Barrier()

        # Initialization of a termination signal
        signal = 0

        mpi_print(f'[runMD]\tGet the initial configuration from {self.MD_input}', rank)
        if self.MD_input == 'start.in':
            # Read the ground state structure with the primitive cell
            struc_init = atoms_read(self.MD_input, format='aims')
            # Make it supercell
            struc = make_supercell(struc_init, self.supercell_init)
            MaxwellBoltzmannDistribution(struc, temperature_K=self.temperature, force_temp=True)
        elif self.MD_input == 'trajectory.son':
            # Read all structural configurations in SON file
            metadata, data = son.load(self.MD_input)
            atom_numbers = [
            atomic_numbers[items[1]]
            for items in data[-1]['atoms']['symbols']
            for jndex in range(items[0])
            ]
            struc_son = Atoms(
                atom_numbers,
                positions=data[-1]['atoms']['positions'],
                cell=data[-1]['atoms']['cell'],
                pbc=data[-1]['atoms']['pbc']
                )
            struc_son.set_velocities(data[-1]['atoms']['velocities'])
            struc = make_supercell(struc_son, self.supercell_init)
        else:
            mpi_print(f'[runMD]\tYou need to assign MD_input appropriately.', rank)
            signal = 1
        comm.Barrier()

        ## Prepare the ground state structure
        # Read the ground state structure with the primitive cell
        struc_init = atoms_read('geometry.in.next_step', format='aims')
        # Make it supercell
        struc_super = make_supercell(struc_init, self.supercell)
        # Get the number of atoms in the simulation cell
        self.NumAtoms = len(struc_super)
        mpi_print(f'[runMD]\tRead the reference structure: geometry.in.next_step', rank)
        comm.Barrier()

        mpi_print(f'[runMD]\tFind the trained models: {self.modelpath}', rank)
        # Prepare empty lists for potential and total energies
        Epot_step = []
        calc_MLIP = []
        for index_nmodel in range(self.nmodel):
            for index_nstep in range(self.nstep):
                if (index_nmodel * self.nstep + index_nstep) % size == rank:
                    dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
                    if os.path.exists(f'{self.modelpath}/{dply_model}'):
                        single_print(f'\t\tFound the deployed model: {dply_model}')
                        calc_MLIP.append(
                            nequip_calculator.NequIPCalculator.from_deployed_model(
                                f'{self.modelpath}/{dply_model}'
                                )
                        )
                        struc_super.calc = calc_MLIP[-1]
                        Epot_step.append(struc_super.get_potential_energy() - self.E_gs)
                    else:
                        # If there is no model, turn on the termination signal
                        single_print(f'\t\tCannot find the model: {dply_model}')
                        signal = 1
                        signal = comm.bcast(signal, root=rank)
        Epot_step = comm.allgather(Epot_step)

        # Get averaged energy from trained models
        Epot_step_avg =\
        np.average(np.array([i for items in Epot_step for i in items]), axis=0)
        mpi_print(f'[runMD]\tGet the potential energy of the reference structure: {self.Epot_step_avg}', rank)

        # if the number of trained model is not enough, terminate it
        if signal == 1:
            sys.exit()
        comm.Barrier()

        mpi_print(f'[runMD]\tInitiate MD with trained models', rank)
        runMD(
            struc, self.ensemble, self.temperature,
            self.pressure, self.timestep, self.friction,
            self.compressibility, self.taut, self.taup,
            self.mask, self.loginterval, self.steps,
            self.nstep, self.nmodel, Epot_step_avg, self.al_type,
            self.logfile, self.trajectory, calc_MLIP,
            signal_uncert=True, signal_append=True
            )

        mpi_print(f'[runMD]\t!! Finish MD calculations', rank)



    def run_dft_cnvg(self):
        """Function [run_dft_cnvg]
        Implement the convergence test with trained models
        """

        # Extract MPI infos
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # Print the head
        output_init('cnvg', self.version, MPI=True)
        mpi_print(f'[cnvg]\tGet the convergence of {self.nmodel}x{self.nstep} matrix', rank)
        comm.Barrier()

        # Specify the path to the test data
        npz_test = f'./MODEL/data-test.npz'
        data_test = np.load(npz_test)
        NumAtom = len(data_test['z'][0])
        mpi_print(f'[cnvg]\tLoad testing data: {npz_test}', rank)
        comm.Barrier()

        # Specify the working path and initialize the signal variable
        workpath = f'./MODEL/{self.temperature}K-{self.pressure}bar_0'
        signal = 0

        mpi_print(f'[cnvg]\tFind the trained models: {workpath}', rank)
        # Load the trained models as calculators
        calc_MLIP = []
        for index_nmodel in range(self.nmodel):
            for index_nstep in range(self.nstep):
                if (index_nmodel * self.nstep + index_nstep) % size == rank:
                    dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
                    if os.path.exists(f'{workpath}/{dply_model}'):
                        mpi_print(f'\t\tFound the deployed model: {dply_model}', rank=0)
                        calc_MLIP.append(
                            nequip_calculator.NequIPCalculator.from_deployed_model(f'{workpath}/{dply_model}')
                        )
                    else:
                        mpi_print(f'\t\tCannot find the model: {dply_model}', rank=0)
                        signal = 1
                        signal = comm.bcast(signal, root=rank)

        # Terminate the code when there is no tained model
        if signal == 1:
            mpi_print('[cnvg]\tNot enough trained models', rank)
            sys.exit()
        comm.Barrier()

        # Predict matrices of energy and forces and their R2 and MAE 
        mpi_print(f'[cnvg]\tGo through all trained models for the testing data', rank)
        zndex = 0
        
        prd_E_total = []
        R2_E_total = []
        MAE_E_total = []
        
        prd_F_total = []
        R2_F_total = []
        MAE_F_total = []

        for index_nmodel in range(self.nmodel):
            for index_nstep in range(self.nstep):
                if (index_nmodel * self.nstep + index_nstep) % size == rank:
                    prd_E = []
                    prd_F = []
                    for idx, (id_R, id_z, id_CELL, id_PBC) in enumerate(zip(
                        data_test['R'], data_test['z'],
                        data_test['CELL'], data_test['PBC']
                    )):
                        struc = Atoms(
                            id_z,
                            positions=id_R,
                            cell=id_CELL,
                            pbc=id_PBC
                        )
                        struc.calc = calc_MLIP[zndex]
                        prd_E.append(struc.get_potential_energy())
                        prd_F.append(struc.get_forces())
                        mpi_print(f'\t\t\tTesting data:{idx}', rank)
                    zndex += 1
                    data_E = np.ndarray.tolist(data_test['E'])
                    data_F = np.ndarray.tolist(data_test['F'])

                    R2_E_total.append({f'{index_nmodel}_{index_nstep}': r2_score(data_test['E'], prd_E)})
                    MAE_E_total.append({f'{index_nmodel}_{index_nstep}': mean_absolute_error(data_test['E'], prd_E)})
                    prd_E_total.append({f'{index_nmodel}_{index_nstep}': np.average(np.array(prd_E))})

                    R2_F_total.append({f'{index_nmodel}_{index_nstep}': r2_score(np.array(data_test['F']).flatten(), np.array(prd_F).flatten())})
                    MAE_F_total.append({f'{index_nmodel}_{index_nstep}': mean_absolute_error(np.array(data_test['F']).flatten(), np.array(prd_F).flatten())})
                    prd_F_total.append({f'{index_nmodel}_{index_nstep}': np.average(np.array(prd_F).flatten())})
                    mpi_print(f'\t\tCollect the prediction: [{index_nmodel},{index_nstep}]', rank=0)

        R2_E_total = comm.allgather(R2_E_total)
        MAE_E_total = comm.allgather(MAE_E_total)
        prd_E_total = comm.allgather(prd_E_total)

        R2_F_total = comm.allgather(R2_F_total)
        MAE_F_total = comm.allgather(MAE_F_total)
        prd_F_total = comm.allgather(prd_F_total)

        
        R2_E_matrix = np.empty([self.nmodel, self.nstep])
        MAE_E_matrix = np.empty([self.nmodel, self.nstep])
        prd_E_matrix = np.empty([self.nmodel, self.nstep])

        R2_F_matrix = np.empty([self.nmodel, self.nstep])
        MAE_F_matrix = np.empty([self.nmodel, self.nstep])
        prd_F_matrix = np.empty([self.nmodel, self.nstep])

        # Due to the nasty MPI operation with trained model due to the memory
        # Here, need to reconstruct the matrices in ordered way
        mpi_print(f'[cnvg]\tBuild the matrices', rank)
        for index_nmodel in range(self.nmodel):
            for index_nstep in range(self.nstep):
                for item in R2_E_total:
                    if item != []:
                        for dict_item in item:
                            if f'{index_nmodel}_{index_nstep}' in dict_item.keys():
                                R2_E_matrix[index_nmodel, index_nstep] = dict_item[f'{index_nmodel}_{index_nstep}']
                                
                for item in MAE_E_total:
                    if item != []:
                        for dict_item in item:
                            if f'{index_nmodel}_{index_nstep}' in dict_item.keys():
                                MAE_E_matrix[index_nmodel, index_nstep] = dict_item[f'{index_nmodel}_{index_nstep}']
                                
                for item in prd_E_total:
                    if item != []:
                        for dict_item in item:
                            if f'{index_nmodel}_{index_nstep}' in dict_item.keys():
                                prd_E_matrix[index_nmodel, index_nstep] = dict_item[f'{index_nmodel}_{index_nstep}']

                for item in R2_F_total:
                    if item != []:
                        for dict_item in item:
                            if f'{index_nmodel}_{index_nstep}' in dict_item.keys():
                                R2_F_matrix[index_nmodel, index_nstep] = dict_item[f'{index_nmodel}_{index_nstep}']
                                
                for item in MAE_F_total:
                    if item != []:
                        for dict_item in item:
                            if f'{index_nmodel}_{index_nstep}' in dict_item.keys():
                                MAE_F_matrix[index_nmodel, index_nstep] = dict_item[f'{index_nmodel}_{index_nstep}']
                                
                for item in prd_F_total:
                    if item != []:
                        for dict_item in item:
                            if f'{index_nmodel}_{index_nstep}' in dict_item.keys():
                                prd_F_matrix[index_nmodel, index_nstep] = dict_item[f'{index_nmodel}_{index_nstep}']


        np.savez('E_matrix_R2', E=R2_E_matrix)
        np.savez('E_matrix_MAE', E=MAE_E_matrix)
        np.savez('E_matrix_prd', E=prd_E_matrix)

        np.savez('F_matrix_R2', E=R2_F_matrix)
        np.savez('F_matrix_MAE', E=MAE_F_matrix)
        np.savez('F_matrix_prd', E=prd_F_matrix)
        mpi_print(f'[cnvg]\tSave matrices: E_matrix and F_matrix', rank)

        prd_Eavg_matrix = np.empty([nmodel, nstep])
        prd_Estd_matrix = np.empty([nmodel, nstep])
        prd_Favg_matrix = np.empty([nmodel, nstep])
        prd_Fstd_matrix = np.empty([nmodel, nstep])

        # Here, get the convergence-testing results averaging over different number of matrix elements
        mpi_print(f'[cnvg]\tGet average with different ranges', rank)
        for index_nmodel in range(nmodel):
            for index_nstep in range(nstep):
                prd_Eavg_matrix[index_nmodel, index_nstep] = np.average(prd_E_matrix[:(index_nmodel+1),:(index_nstep+1)])
                prd_Estd_matrix[index_nmodel, index_nstep] = np.std(prd_E_matrix[:(index_nmodel+1),:(index_nstep+1)])

                prd_Favg_matrix[index_nmodel, index_nstep] = np.average(prd_F_matrix[:(index_nmodel+1),:(index_nstep+1)])
                prd_Fstd_matrix[index_nmodel, index_nstep] = np.std(prd_F_matrix[:(index_nmodel+1),:(index_nstep+1)])

        np.savez('E_converge_avg', E=prd_Eavg_matrix)
        np.savez('E_converge_std', E=prd_Estd_matrix)
        np.savez('F_converge_avg', E=prd_Favg_matrix)
        np.savez('F_converge_std', E=prd_Fstd_matrix)

        mpi_print(f'[cnvg]\tSave convergence results: E_converge and F_converge', rank)

