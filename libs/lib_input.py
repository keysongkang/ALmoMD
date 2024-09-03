from libs.lib_util import read_input_file

class inputs:
    """Object [almd]
    Read, update, and store input variables.

    Functions:

    __init__: Automatically initialize the inputs and read input.in file.
    """

    def __init__(self, input_file='input.in'):
        """Function [__init__]
        Automatically initialize the inputs and read input.in file.

        Parameters:

        input_file: Read the inputs from input.in file 
        """

        ### Version
        self.version = '0.2.0'

        ### Default setting for inputs
        ##[Active learning types]
        # calc_type: str
        #     Type of sampling; 'active' (active learning), 'random', 'converge'
        self.calc_type = 'active'
        # al_type: str
        #     Type of active learning; 'energy', 'force', 'force_max'
        self.al_type = 'force_max'
        # uncert_type: str
        #     Type of uncertainty; 'absolute', 'relative'
        self.uncert_type = 'absolute'
        # uncert_shift: boolean
        #     Shifting of erf function
        self.harmonic_F = False
        self.anharmonic_F = False
        # uncert_type: str
        #     Use the harmonic force constants
        self.uncert_shift = 2.0
        # uncert_grad: float
        #     Gradient of erf function
        #     (Value is relative to standard deviation)
        self.uncert_grad = 1.0
        # output_format: str
        #     Type of FHI-vibes output to be read
        self.output_format = 'trajectory.son'

        ##[Active learning setting]

        self.MLIP = 'nequip'

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
        self.ntrain_init = 25
        # ntrain: int
        #     The number of added training data for each iterative step
        self.ntrain = 25
        # nperiod: int
        #     The number of steps for the active learning with a finite period
        self.nperiod = 200
        # crtria_cnvg: float
        #     Convergence criteria
        self.crtria_cnvg = 0.0000001
        # num_calc: int
        #     The number of job scripts for DFT calculations
        self.num_calc = 16
        # num_mdl_calc: int
        #     The number of job scripts for MLIP training calculations
        self.num_mdl_calc = 6
        # random_index: int
        #     The number of iteration for DFT random sampling
        self.random_index = 1

        self.loss_var = True
        self.train_stress = False
        self.npz_sigma = True

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
        self.printinterval = 1
        # temperature: float
        #     The desired temperature in units of Kelvin (K)
        self.temperature = 2000
        # pressure: float
        #     The desired pressure in units of eV/Angstrom**3 (atomic unit)
        self.pressure = 0

        self.ttime = 500   # [fs]
        self.pfactor = None

        self.signal_uncert = True

        self.meta_Ediff = 0.0
        self.meta_restart = False
        self.meta_r_crtria = 100.0

        self.idx_atom = 0 
        self.bias_A = 0.0
        self.bias_B = 999

        self.temp_factor = 0.0

        # Set the testing data
        # test / train
        self.npz_test = 'test'

        # Usage of energy criteria
        self.criteria_energy = True

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
        # supercell_init: 2D array of floats
        #     Tensor shape of supercell for runMD calculations
        #     compared to primitive cell.
        self.supercell_init = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.MD_search = 'continue'
        self.cell_factor = 1.0

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

        ##[so3krates setting]
        self.r_cut = 5.0
        self.l = 3
        self.f = 132
        self.l_min = 1
        self.l_max = 3
        self.max_body_order = 2
        self.f_body_order = 1
        self.we = 0.01 # weight-energy
        self.wf = 1.00 # weight-forces
        self.ws = 'None' # weight-stress
        self.loss_variance_scaling = '--no-loss-variance-scaling'
        self.epochs = 2000
        self.eval_energy_t = 'None'
        self.mic = '--mic'
        self.float64 = '--no-float64'
        self.lr = 0.001
        self.lr_stop = 0.00001
        self.lr_decay_exp_transition_steps = 100000
        self.lr_decay_exp_decay_factor = 0.7
        self.clip_by_global_norm = 'None'
        self.shift_mean = '--shift-mean'
        self.size_batch = 'None'
        self.size_batch_training = 'None'
        self.size_batch_validation = 'None'
        # self.seed_model = 0
        self.seed_data = 0
        self.seed_training = 0
        self.wandb_name = 'None'
        self.wandb_group = 'None'
        self.wandb_project = 'None'
        # self.outfile_inputs = 'inputs.json'
        self.overwrite_module = '--no-overwrite-module'
        self.ace = '--no-ace'
        self.skin = 0.01

        ##[Constants]
        # kB: float
        #     Boltzmann constant in units of eV/K
        self.kB = 8.617333262e-5

        self.job_command = 'sbatch'
        self.job_dft_name = 'job-vibes.slurm'
        self.job_MLIP_name = 'job-nequip-gpu.slurm'
        # Prepare the command line for FHI-vibes
        self.vibes_command = 'vibes run singlepoint aims.in &> log.aims'

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

        self.train_split = self.ntrain/(self.ntrain + self.nval)

        ### Initization of parameters
        # index: int
        #     The index of AL interactive steps
        self.index = 0


        # # Extract MPI infos
        # from mpi4py import MPI

        # self.comm = MPI.COMM_WORLD
        # self.size = self.comm.Get_size()
        # self.rank = self.comm.Get_rank()
