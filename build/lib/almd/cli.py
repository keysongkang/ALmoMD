import argparse
import argcomplete
from scripts.utils import aims2son, split_son


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


def aims2son_command(args):
    """
    Implement the logic for "almomd utils aims2son" command
    This function will handle the conversion from aims to son format

    Parameters:

    args: class
        Contains inputs from the command line
    args.temperature: float
        temperature (K)
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

    # Subparser for "utils" command
    utils_parser = subparsers.add_parser(
        'utils',
        help='Load the necessary utilities'
        )
    utils_subparsers = utils_parser.add_subparsers()

    # Subparser for "utils aims2son" subcommand
    aims2son_parser = utils_subparsers.add_parser(
        'aims2son',
        help="Convert aims.out to trajectory.son"
             "(Temperature is used for assigning atom velocities, "
             "utilizing the Maxwell-Boltzmann distribution)"
        )
    aims2son_parser.add_argument(
        'temperature', nargs='?', type=int,
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
