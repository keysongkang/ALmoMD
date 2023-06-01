import argparse
import argcomplete
from scripts.utils import aims2son, split_son, harmonic_run, harmonic2son


def init_command(args):
    """
    Implement the logic for "almomd init" command
    """
    from scripts.almd import almd
    run_almd = almd()
    run_almd.run_dft_init()


def cont_command(args):
    """
    Implement the logic for "almomd cont" command
    """
    from scripts.almd import almd
    run_almd = almd()
    run_almd.run_dft_cont()


def gen_command(args):
    """
    Implement the logic for "almomd gene" command
    """
    from scripts.almd import almd
    run_almd = almd()
    run_almd.run_dft_gen()


def aiMD_rand_command(args):
    """
    Implement the logic for "almomd aimd-random" command
    """
    from scripts.almd import almd
    run_almd = almd()
    run_almd.run_dft_rand()


def test_command(args):
    """
    Implement the logic for "almomd test" command
    """
    from scripts.almd import almd
    run_almd = almd()
    run_almd.run_dft_test()


def runmd_command(args):
    """
    Implement the logic for "almomd runMD" command
    """
    from scripts.almd import almd
    run_almd = almd()
    run_almd.run_dft_runmd()


def aims2son_command(args):
    """
    Implement the logic for "almomd utils aims2son" command
    This function will handle the conversion from aims to son format

    Parameters:

    args: class
        Contains inputs from the command line
    args.temperature: float
        Temperature (K)
    """
    if not hasattr(args, 'temperature') or args.temperature is None:
        print('Please provide the simulated temperature.')
    else:
        aims2son(args.temperature)


def split_son_command(args):
    """
    Implement the logic for "almomd utils split" command

    Parameters:

    args: class
        Contains inputs from the command line
    args.num_split: int
        The number of testing data
    args.E_gs: float
        Value of E_gs
    """
    if not hasattr(args, 'num_split') or args.num_split is None or \
            not hasattr(args, 'E_gs') or args.E_gs is None:
        print('Please provide the number of test data or a reference energy. '
              '(e.g. almomd utils split 600 -120.30)')
    else:
        split_son(args.num_split, args.E_gs)


def harmonic_run_command(args):
    """
    Implement the logic for "almomd utils harmonic_run" command

    Parameters:

    args: class
        Contains inputs from the command line
    args.temperature: float
        Temperature (K)
    args.num_sample: int
        The number of harmonic samples
    args.num_calc: int
        The number of job scripts to be submitted
    """
    if not hasattr(args, 'temperature') or args.temperature is None or \
            not hasattr(args, 'num_sample') or args.num_sample is None or \
            not hasattr(args, 'DFT_calc') or args.DFT_calc is None or \
            not hasattr(args, 'num_calc') or args.num_calc is None:
        print(
            'Please provide the temperature, the number of harmoic samples'
            ', DFT calculator and the number of job scripts to be submitted.'
            '(e.g. almomd utils harmonic_run 300 10 aims 3)'
            )
    else:
        harmonic_run(args.temperature, args.num_sample, args.DFT_calc, args.num_calc)


def harmonic2son_command(args):
    """
    Implement the logic for "almomd utils harmonic2son" command

    Parameters:

    args: class
        Contains inputs from the command line
    args.temperature: float
        Temperature (K)
    args.num_sample: int
        The number of harmonic samples
    """
    if not hasattr(args, 'temperature') or args.temperature is None or \
            not hasattr(args, 'num_sample') or args.num_sample is None:
        print(
            'Please provide the temperature and the number of harmoic samples'
            '(e.g. almomd utils harmonic2son 300 10)'
            )
    else:
        harmonic2son(args.temperature, args.num_sample)


def traj_run_command(args):
    """
    Implement the logic for "almomd utils harmonic_run" command

    Parameters:

    args: class
        Contains inputs from the command line
    args.traj_path: str
        Path to the trajectory file
    args.thermal_cutoff: int
        Thermalization cutoff
    args.num_traj: int
        The number of configurations to be calculated by DFT
    args.DFT_calc: str
        The name of the DFT calculator
    """
    if not hasattr(args, 'traj_path') or args.traj_path is None or \
            not hasattr(args, 'thermal_cutoff') or args.thermal_cutoff is None or \
            not hasattr(args, 'num_traj') or args.num_traj is None or \
            not hasattr(args, 'DFT_calc') or args.DFT_calc is None:
        print(
            'Please provide the path to the trajectory file, '
            'thermalization cutoff, the number of configurations '
            'to be calculated by DFT and the DFT calclulator'
            '(e.g. almomd utils traj_run md.traj 300 500 aims)'
            )
    else:
        harmonic_run(args.traj_path, args.thermal_cutoff, args.num_traj, args.DFT_calc)


def main():
    """
    Main parser for "ALmoMD" command
    """

    parser = argparse.ArgumentParser(prog='almomd')
    subparsers = parser.add_subparsers()

    # Subparser for "initiation" command
    initiation_parser = subparsers.add_parser(
        'init',
        help='Performing the calculation by initializing with the '
             'aiMD trajectory'
    )
    initiation_parser.set_defaults(func=init_command)

    # Subparser for "continuation" command
    continuation_parser = subparsers.add_parser(
        'cont',
        help='Running and Continuing the ALMD calculation'
        )
    continuation_parser.set_defaults(func=cont_command)

    # Subparser for "generation" command
    generation_parser = subparsers.add_parser(
        'gen',
        help='Extract DFT results, generate the training data, '
             'and execute NequIP'
        )
    generation_parser.set_defaults(func=gen_command)

    # Subparser for "aiMD-random" command
    generation_parser = subparsers.add_parser(
        'aiMD-rand',
        help='Implement the ALMD calculation using random samplings'
        )
    generation_parser.set_defaults(func=aiMD_rand_command)

    # Subparser for "test" command
    test_parser = subparsers.add_parser(
        'test',
        help='Check the validation error'
        )
    test_parser.set_defaults(func=test_command)

    # Subparser for "runMD" command
    runmd_parser = subparsers.add_parser(
        'runMD',
        help='Initiate MD calculation using trained models'
        )
    runmd_parser.set_defaults(func=runmd_command)

    # Subparser for "utils" command
    utils_parser = subparsers.add_parser(
        'utils',
        help='Load the necessary utilities'
        )
    utils_subparsers = utils_parser.add_subparsers()

    # Subparser for "utils aims2son" subcommand
    aims2son_parser = utils_subparsers.add_parser(
        'aims2son',
        help="Convert aims.out to trajectory.son "
             "(Temperature is used for assigning atom velocities, "
             "utilizing the Maxwell-Boltzmann distribution)"
        )
    aims2son_parser.add_argument(
        'temperature', nargs='?', type=float,
        help='Temperature (K)'
        )
    aims2son_parser.set_defaults(func=aims2son_command)

    # Subparser for "utils split" subcommand
    split_parser = utils_subparsers.add_parser(
        'split',
        help='Separate trajectory.son into '
             'trajectory_test.son and trajectory_train.son'
        )
    split_parser.add_argument(
        'num_split', nargs='?', type=int,
        help='The number of testing data'
        )
    split_parser.add_argument(
        'E_gs', nargs='?', type=float,
        help='Reference total energy in units of eV/Unitcell'
        )
    split_parser.set_defaults(func=split_son_command)

    # Subparser for "utils harmonic_run" subcommand
    harmonic_run_parser = utils_subparsers.add_parser(
        'harmonic_run',
        help='Initiate FHI-aims and FHI-vibes with '
             'structural configurations from a harmonic sampling'
        )
    harmonic_run_parser.add_argument(
        'temperature', nargs='?', type=float,
        help='Temperature (K)'
        )
    harmonic_run_parser.add_argument(
        'num_sample', nargs='?', type=int,
        help='The number of harmonic samples'
        )
    harmonic_run_parser.add_argument(
        'DFT_calc', nargs='?', type=str,
        help='The name of the DFT calculator'
        )
    harmonic_run_parser.add_argument(
        'num_calc', nargs='?', type=int,
        help='The number of job scripts to be submitted'
        )
    harmonic_run_parser.set_defaults(func=harmonic_run_command)

    # Subparser for "utils harmonic2son" subcommand
    harmonic2son_parser = utils_subparsers.add_parser(
        'harmonic2son',
        help='Collect all results of FHI-vibes calculations '
             'with harmonic samplings and convert it to SON file'
        )
    harmonic2son_parser.add_argument(
        'temperature', nargs='?', type=float,
        help='Temperature (K)'
        )
    harmonic2son_parser.add_argument(
        'num_sample', nargs='?', type=int,
        help='The number of harmonic samples'
        )
    harmonic2son_parser.set_defaults(func=harmonic2son_command)

    # Subparser for "utils harmonic_run" subcommand
    traj_run_parser = utils_subparsers.add_parser(
        'traj_run',
        help='Initiate FHI-aims or FHI-vibes for configurations '
             'from a trajectory file'
        )
    traj_run_parser.add_argument(
        'traj_path', nargs='?', type=str,
        help='Path to the trajectory file'
        )
    traj_run_parser.add_argument(
        'thermal_cutoff', nargs='?', type=int,
        help='Thermalization cutoff'
        )
    traj_run_parser.add_argument(
        'num_traj', nargs='?', type=int,
        help='The number of configurations to be calculated by DFT'
        )
    traj_run_parser.add_argument(
        'DFT_calc', nargs='?', type=str,
        help='The name of the DFT calculator'
        )
    traj_run_parser.set_defaults(func=traj_run_command)

    args = parser.parse_args()

    if hasattr(args, 'func'):
        args.func(args)


if __name__ == '__main__':
    # Enable tab completion [Install instructions]
    # pip install argcomplete
    # eval "$(register-python-argcomplete almomd)"
    # source ~/.bashrc
    argcomplete.autocomplete(parser)

    main()
