from dataclasses import InitVar, dataclass, field, fields

from libs.lib_util import read_input_file


@dataclass
class inputs:
    """Configuration loaded from input.in.

    The lowercase class name is kept for backward compatibility with the
    existing CLI and workflow code, which call ``inputs()`` directly.
    """

    input_file: InitVar[str] = 'input.in'

    version: str = '0.2.0'

    # Active learning
    calc_type: str = 'active'
    al_type: str = 'force_max'
    uncert_type: str = 'absolute'
    harmonic_F: bool = False
    anharmonic_F: bool = False
    uncert_shift: float = 2.0
    uncert_grad: float = 1.0
    output_format: str = 'trajectory.son'
    MLIP: str = 'nequip'

    nstep: int = 1
    nmodel: int = 1
    steps_init: int = 125
    steps_random: int = 125
    ntrain_init: int = 25
    ntrain: int = 25
    nperiod: int = 200
    crtria_cnvg: float = 1.0e-7
    num_calc: int = 16
    num_mdl_calc: int = 6
    random_index: int = 1

    loss_var: bool = True
    train_stress: bool = False
    npz_sigma: bool = True

    # Molecular dynamics
    ensemble: str = 'NVTLangevin'
    supercell: list = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    timestep: float = 1.0
    loginterval: int = 10
    printinterval: int = 1
    temperature: int = 2000
    pressure: int = 0
    ttime: float = 500.0
    pfactor: float = None
    signal_uncert: bool = True

    meta_Ediff: float = 0.0
    meta_restart: bool = False
    meta_r_crtria: float = 100.0
    idx_atom: object = 0
    bias_A: float = 0.0
    bias_B: float = 999.0
    temp_factor: float = 0.0

    npz_test: str = 'test'
    criteria_energy: bool = True

    # Standalone runMD
    modelpath: str = './MODEL'
    logfile: str = 'md.log'
    trajectory: str = 'md.traj'
    steps: int = 500
    supercell_init: list = field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    MD_search: str = 'continue'
    cell_factor: float = 1.0

    # Thermostats/barostats
    friction: float = 0.02
    taut: int = 50
    compressibility: float = 4.57e-5
    taup: int = 100
    mask: tuple = (1, 1, 1)

    # NequIP
    rmax: float = 3.5
    lmax: int = 2
    nfeatures: int = 16
    E_gs: float = 0.0

    # SO3krates
    r_cut: float = 5.0
    l: int = 3
    f: int = 132
    l_min: int = 1
    l_max: int = 3
    max_body_order: int = 2
    f_body_order: int = 1
    we: float = 0.01
    wf: float = 1.0
    ws: str = 'None'
    loss_variance_scaling: str = '--no-loss-variance-scaling'
    epochs: int = 2000
    eval_energy_t: str = 'None'
    mic: str = '--mic'
    float64: str = '--no-float64'
    lr: float = 0.001
    lr_stop: float = 0.00001
    lr_decay_exp_transition_steps: int = 100000
    lr_decay_exp_decay_factor: float = 0.7
    clip_by_global_norm: str = 'None'
    shift_mean: str = '--shift-mean'
    size_batch: object = 'None'
    size_batch_training: object = 'None'
    size_batch_validation: object = 'None'
    seed_data: int = 0
    seed_training: int = 0
    wandb_name: str = 'None'
    wandb_group: str = 'None'
    wandb_project: str = 'None'
    overwrite_module: str = '--no-overwrite-module'
    ace: str = '--no-ace'
    skin: float = 0.01

    # Constants and job commands
    kB: float = 8.617333262e-5
    job_command: str = 'sbatch'
    job_dft_name: str = 'job-vibes.slurm'
    job_MLIP_name: str = 'job-nequip-gpu.slurm'
    vibes_command: str = 'vibes run singlepoint aims.in &> log.aims'
    device: str = 'cpu'
    test_index: int = 1

    # Legacy/rare input keys kept so old input.in files fail less often.
    steps_ther: int = 0
    cutoff_ther: int = 0
    wndex: int = 0

    # Runtime attributes filled by workflows.
    index: int = field(init=False, default=0)
    nval_init: int = field(init=False)
    nval: int = field(init=False)
    ntotal_init: int = field(init=False)
    ntotal: int = field(init=False)
    train_split: float = field(init=False)
    NumAtoms: int = field(init=False, default=0)
    calc_MLIP: object = field(init=False, default=None)

    def __post_init__(self, input_file):
        input_variables = read_input_file(input_file)
        self._apply_input_variables(input_variables)
        self._derive_values()
        self._validate()

    def _apply_input_variables(self, input_variables):
        valid_names = {item.name for item in fields(self) if item.init}
        valid_names.discard('input_file')

        if 'anharmoic_F' in input_variables:
            input_variables['anharmonic_F'] = input_variables.pop('anharmoic_F')

        unknown = sorted(set(input_variables) - valid_names)
        if unknown:
            raise ValueError(
                'Unknown input key(s): '
                + ', '.join(unknown)
                + '. Check input.in for typos or add the key to libs/lib_input.py.'
            )

        for name, value in input_variables.items():
            setattr(self, name, value)

    def _derive_values(self):
        self.nval_init = int(self.ntrain_init / 5)
        self.nval = int(self.ntrain / 5)
        self.ntotal_init = (self.ntrain_init + self.nval_init) * self.nstep
        self.ntotal = (self.ntrain + self.nval) * self.nstep
        self.train_split = self.ntrain / (self.ntrain + self.nval)
        self.index = 0

    def _validate(self):
        _validate_choice(self.calc_type, {'active', 'random', 'period', 'converge'}, 'calc_type')
        _validate_choice(
            self.al_type,
            {'energy', 'energy_max', 'force', 'force_max', 'sigma', 'sigma_max', 'all', 'EorFmax'},
            'al_type',
        )
        _validate_choice(self.uncert_type, {'absolute', 'relative'}, 'uncert_type')
        _validate_choice(self.MLIP, {'nequip', 'so3krates'}, 'MLIP')

        positive_ints = {
            'nstep': self.nstep,
            'nmodel': self.nmodel,
            'ntrain_init': self.ntrain_init,
            'ntrain': self.ntrain,
            'loginterval': self.loginterval,
            'num_calc': self.num_calc,
            'num_mdl_calc': self.num_mdl_calc,
        }
        for name, value in positive_ints.items():
            if int(value) <= 0:
                raise ValueError(f'{name} must be positive, got {value!r}')

        if self.ntrain + self.nval <= 0:
            raise ValueError('ntrain + nval must be positive.')


def _validate_choice(value, valid_values, name):
    if value not in valid_values:
        raise ValueError(
            f'{name} must be one of {sorted(valid_values)}, got {value!r}'
        )


ALmoMDConfig = inputs
