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

from libs.lib_util import check_mkdir, job_dependency, read_input_file, mpi_print
from libs.lib_npz import generate_npz_DFT_init, generate_npz_DFT, generate_npz_DFT_rand_init, generate_npz_DFT_rand
from libs.lib_train import get_train_job, execute_train_job
from libs.lib_dft import run_DFT
from libs.lib_progress    import check_progress, check_index
from libs.lib_mainloop    import MLMD_initial, MLMD_main, MLMD_random
from libs.lib_criteria    import get_criteria
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

        ##[Molecular dynamics]
        # ensemble: str
        #     Type of MD ensembles; 'NVTLangevin'
        self.ensemble = 'NVTLangevin'
        # supercell: 2D array of floats
        #     Tensor shape of supercell for MD calculations
        #     compared to primitive cell.
        self.supercell = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
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

        # Read aiMD trajectory file of training data
        metadata, traj = son.load('trajectory_train.son')

        # Set the path to folders storing the training data for NequIP
        workpath = f'./data/{self.temperature}K-{self.pressure}bar_{self.index}'
        # Create these folders
        if rank == 0:
            check_mkdir('data')
            check_mkdir(workpath)
        comm.Barrier()

        # As it is an initialization step,
        # the total number of training and validation data matches the initial settings
        total_ntrain = self.ntrain_init
        total_nval = self.nval_init

        # Generate first set of training data in npz files from trajectory file
        if rank == 0:
            generate_npz_DFT_init(
                traj, self.ntrain_init, self.nval_init, self.nstep, self.E_gs, self.index,
                self.temperature, self.pressure, workpath
            )
        del traj  # Remove it to reduce the memory usage
        comm.Barrier()

        # Training process: Run NequIP
        execute_train_job(
            total_ntrain, total_nval, self.rmax, self.lmax, self.nfeatures,
            workpath, self.nstep, self.nmodel
        )

        # Submit a job-dependence to execute run_dft_cont after the NequIP training
        # For 'converge' setting, we don't needo to submit it.
        if not self.calc_type == 'converge':
            if rank == 0:
                job_dependency('cont')
    


    def run_dft_cont(self):
        """Function [run_dft_cont]
        Run and continue the ALMD calculation.
        """

        # Extract MPI infos
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        ## Prepare the ground state structure
        # Read the ground state structure with the primitive cell
        struc_init = atoms_read('geometry.in.next_step', format='aims')
        # Make it supercell
        struc_relaxed = make_supercell(struc_init, self.supercell)
        # Get the number of atoms in the simulation cell
        self.NumAtoms = len(struc_relaxed)

        ### Initizlization step
        ## For active learning sampling,
        if self.calc_type == 'active':
            # Retrieve the calculation index (kndex: MLMD_initial, MD_index: MLMD_main, signal: termination)
            # to resume the MLMD calculation if a previous one exists.
            kndex, MD_index, signal = None, None, None
            if rank == 0:
                # Open the uncertainty output file
                kndex, MD_index, self.index, signal = check_progress(
                    self.temperature, self.pressure, self.ensemble, self.timestep,
                    self.friction, self.compressibility, self.taut, self.taup, self.mask,
                    self.loginterval, self.name, self.supercell,
                    self.ntotal, self.ntrain, self.ntrain_init, self.nval, self.nval_init, self.rmax,
                    self.lmax, self.nfeatures, self.nstep, self.nmodel, self.steps_init, self.index,
                    self.crtria, self.crtria_cnvg, self.NumAtoms
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
            total_index = None
            if rank == 0:
                self.index = check_index(self.index)
            self.index = comm.bcast(self.index, root=0)

        # If we get the signal from check_progress, the script will be terminated.
        if signal == 1:
            sys.exit()

        ##!! We need to check whether this one is needed or not.
        # Get total_index to resume the MLMD calculation
        total_index = None
        if rank == 0:
            # Open the result.txt file
            total_index = check_index(self.index)
        total_index = comm.bcast(total_index, root=0)

        # Get the total number of traning and validating data at current step
        self.total_ntrain = self.ntrain * total_index + self.ntrain_init
        self.total_nval = self.nval * total_index + self.nval_init


        ### Get calculators
        # Set the path to folders finding the trained model from NequIP
        workpath = f'./data/{self.temperature}K-{self.pressure}bar_{self.index}'
        # Create these folders
        if rank == 0:
            check_mkdir('data')
            check_mkdir(workpath)
        comm.Barrier()

        # Get calculators from previously trained MLIP and its total energy of ground state structure
        E_ref, calc_MLIP = get_train_job(
            struc_relaxed, self.total_ntrain, self.total_nval, self.rmax, self.lmax,
            self.nfeatures, workpath, self.nstep, self.nmodel
        )


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
                self.compressibility, self.taut, self.taup, self.mask, self.loginterval, self.steps_init * self.loginterval,
                self.nstep, self.nmodel, calc_MLIP, E_ref
            )
        else:
            raise ValueError("Invalid calc_type. Supported values are 'active' and 'random'.")

        # Submit job-scripts for DFT calculations with sampled configurations and job-dependence for run_dft_gen
        if rank == 0:
            run_DFT(self.temperature, self.pressure, self.index, (self.ntrain + self.nval) * self.nstep)



    def run_dft_gen(self):
        """Function [run_dft_gen]
        Extract DFT results, generate the training data, and execute NequIP.
        """

        # Extract MPI infos
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        ## Prepare the ground state structure
        # Read the ground state structure with the primitive cell
        struc_init = atoms_read('geometry.in.next_step', format='aims')
        # Make it supercell
        struc_relaxed = make_supercell(struc_init, self.supercell)
        # Get the number of atoms in unitcell
        self.NumAtoms = len(struc_relaxed)

        ### Initizlization step
        ## For active learning sampling,
        if self.calc_type == 'active':
            # Retrieve the calculation index (kndex: MLMD_initial, MD_index: MLMD_main, signal: termination)
            # to resume the MLMD calculation if a previous one exists.
            kndex, MD_index, signal = None, None, None
            if rank == 0:
                # Open the uncertainty output file
                kndex, MD_index, self.index, signal = check_progress(
                    self.temperature, self.pressure, self.ensemble, self.timestep,
                    self.friction, self.compressibility, self.taut, self.taup, self.mask,
                    self.loginterval, self.name, self.supercell,
                    self.ntotal, self.ntrain, self.ntrain_init, self.nval, self.nval_init, self.rmax,
                    self.lmax, self.nfeatures, self.nstep, self.nmodel, self.steps_init, self.index,
                    self.crtria, self.crtria_cnvg, self.NumAtoms
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
            sys.exit()

        ##!! We need to check whether this one is needed or not.
        # Get total_index to resume the MLMD calculation
        total_index = None
        if rank == 0:
            total_index = check_index(self.index)
        total_index = comm.bcast(total_index, root=0)

        # Get the total number of traning and validating data at current step
        self.total_ntrain = self.ntrain * total_index + self.ntrain_init
        self.total_nval = self.nval * total_index + self.nval_init


        ### Get calculators
        # Set the path to folders storing the training data for NequIP
        workpath = f'./data/{self.temperature}K-{self.pressure}bar_{self.index}'
        if rank == 0:
            check_mkdir('data')
            check_mkdir(workpath)
        comm.Barrier()

        # Generate first set of training data in npz files from trajectory file
        if rank == 0:
            generate_npz_DFT(
                self.ntrain, self.nval, self.nstep, self.E_gs, self.index, self.temperature,
                self.output_format, self.pressure, workpath
            )

        # Training process: Run NequIP
        execute_train_job(
            self.total_ntrain, self.total_nval, self.rmax, self.lmax, self.nfeatures,
            workpath, self.nstep, self.nmodel
        )

        # Check the termination signal
        if signal == 1:
            mpi_print(f'{self.temperature}K is converged.')
            sys.exit()

        # Submit a job-dependence to execute run_dft_cont after the NequIP training
        if rank == 0:
            job_dependency('cont')



    def run_dft_rand(self):
        """Function [run_dft_rand]
        Implement the ALMD calculation using random samplings.
        """

        # Extract MPI infos
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        # Read aiMD trajectory file of training data
        metadata, traj = son.load('trajectory_train.son')

        # As it is an initialization step,
        # the total number of training and validation data matches the initial settings
        total_ntrain = self.ntrain_init
        total_nval = self.nval_init

        # Start from the first step
        index = 0
        # Set the path to folders storing the training data for NequIP
        workpath = f'./data/{self.temperature}K-{self.pressure}bar_{index}'

        if rank == 0:
            # Create these folders
            check_mkdir('data')
            check_mkdir(workpath)

            # Generate first set of training data in npz files from trajectory file
            traj_idx = generate_npz_DFT_rand_init(
                traj, self.ntrain_init, self.nval_init, self.nstep, self.E_gs, index, self.temperature,
                self.pressure, workpath
            )
        comm.Barrier()

        # Training process: Run NequIP
        execute_train_job(
            total_ntrain, total_nval, self.rmax, self.lmax, self.nfeatures,
            workpath, self.nstep, self.nmodel
        )

        # Run steps until random_index (which is assigned in input.in)
        while index < self.random_index:
            # Go to the next step
            index += 1

            # Add the number of new training and validating data
            total_ntrain += self.ntrain
            total_nval += self.nval

            # Set the path to folders storing the training data for NequIP at the current step
            workpath = f'./data/{self.temperature}K-{self.pressure}bar_{index}'

            if rank == 0:
                # Create these folders
                check_mkdir(workpath)

                # Generate first set of training data in npz files from trajectory file
                traj_idx = generate_npz_DFT_rand(
                    traj, self.ntrain, self.nval, self.nstep, self.E_gs, index, self.temperature,
                    self.pressure, workpath, traj_idx
                )
            comm.Barrier()

            # Training process: Run NequIP
            execute_train_job(
                total_ntrain, total_nval, self.rmax, self.lmax, self.nfeatures,
                workpath, self.nstep, self.nmodel
            )


    def run_dft_test(self):
        """Function [run_dft_test]
        Check the validation error.
        """

        # Extract MPI infos
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        # Read testing data
        npz_test = f'./../data/data-test.npz' # Path
        data_test = np.load(npz_test)         # Load
        NumAtom = len(data_test['z'][0])      # Get the number of atoms in the simulation cell

        # Set the path to folders storing the training data for NequIP
        workpath = f'./../data/{self.temperature}K-{self.pressure}bar_{self.wndex}'
        # Initialization of a termination signal
        signal = 0

        # Load the trained models as calculators
        calc_MLIP = []
        for index_nmodel in range(self.nmodel):
            for index_nstep in range(self.nstep):
                if (index_nmodel * self.nstep + index_nstep) % size == rank:
                    dply_model = f'deployed-model_{index_nmodel}_{index_nstep}.pth'
                    if os.path.exists(f'{workpath}/{dply_model}'):
                        mpi_print(f'Found the deployed model: {dply_model}', rank)
                        calc_MLIP.append(
                            nequip_calculator.NequIPCalculator.from_deployed_model(f'{workpath}/{dply_model}')
                        )
                    else:
                        # If there is no model, turn on the termination signal
                        mpi_print(f'Cannot find the model: {dply_model}', rank)
                        signal = 1
                        signal = comm.bcast(signal, root=rank)

        # Check the termination signal
        if signal == 1:
            mpi_print('Some trained models are not finished.', rank)
            sys.exit()

        # Open the file to store the results
        if rank == 0:
            outputfile = open(f'result-test_{self.wndex}.txt', 'w')
            outputfile.write('index   \tUncertAbs\tRealErrorAbs\tRealE\tPredictE\n')
            outputfile.close()

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
                        # Predicted energy is shifted E_gs back, but the E_gs defualt is zero
                        prd_E.append(struc.get_potential_energy() + self.E_gs)
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
                trajfile = open(f'result-test_{self.wndex}_energy.txt', 'a')
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
                trajfile = open(f'result-test_{self.wndex}_force.txt', 'a')
                for kndex in range(len(prd_F_avg)):
                    for lndex in range(3):
                        trajfile.write(
                            '{:.10e}'.format(Decimal(str(id_F[kndex][lndex]))) + '\t' +     # Real force
                            '{:.10e}'.format(Decimal(str(prd_F_avg[kndex][lndex]))) + '\n'  # Predicted force
                        )
                trajfile.close()

            config_idx += 1


        ## Plot the energy and force prediction results
        # Read the energy data
        data = pd.read_csv(f'result-test_{self.wndex}_energy.txt', sep="\t")

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
        data = pd.read_csv(f'result-test_{self.wndex}_force.txt', sep="\t", header=None)

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
        mpi_print(f'Energy: {R2_E}\t{MAE_E}\t{R2_F}\t{MAE_F}')
