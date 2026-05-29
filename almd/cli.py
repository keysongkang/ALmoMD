import argparse


def _run_with_inputs(runner):
    from libs.lib_input import inputs
    runner(inputs())


def init_command(args):
    from scripts.lib_run_dft_init import run_dft_init
    _run_with_inputs(run_dft_init)


def cont_command(args):
    from scripts.lib_run_dft_cont import run_dft_cont
    _run_with_inputs(run_dft_cont)


def gen_command(args):
    from scripts.lib_run_dft_gen import run_dft_gen
    _run_with_inputs(run_dft_gen)


def aiMD_rand_command(args):
    from scripts.lib_run_dft_rand import run_dft_rand
    _run_with_inputs(run_dft_rand)


def test_command(args):
    from scripts.lib_run_dft_test import run_dft_test
    _run_with_inputs(run_dft_test)


def runmd_command(args):
    from scripts.lib_run_dft_runmd import run_dft_runmd
    _run_with_inputs(run_dft_runmd)


def cnvg_command(args):
    from scripts.lib_run_dft_cnvg import run_dft_cnvg
    _run_with_inputs(run_dft_cnvg)


def aims2son_command(args):
    from scripts.utils import aims2son
    aims2son(args.temperature)


def split_son_command(args):
    from scripts.utils import split_son
    split_son(args.num_split, args.E_gs, args.harmonic_F)


def harmonic_run_command(args):
    from scripts.utils import harmonic_run
    harmonic_run(args.temperature, args.num_sample, args.DFT_calc, args.num_calc)


def harmonic2son_command(args):
    from scripts.utils import harmonic2son
    harmonic2son(args.temperature, args.num_sample, args.output_format)


def traj_run_command(args):
    from scripts.utils import traj_run
    traj_run(
        args.traj_path,
        args.thermal_cutoff,
        args.num_traj,
        args.DFT_calc,
        args.num_calc,
        args.harmonic_F,
    )


def cnvg_post_command(args):
    from scripts.utils import cnvg_post
    cnvg_post(args.nmodel, args.nstep)


def convert_npz_command(args):
    from scripts.utils import convert_npz
    convert_npz(args.name, args.harmonic_F)


def _add_root_commands(subparsers):
    commands = [
        ('init', 'Perform the calculation by initializing with the aiMD trajectory', init_command),
        ('cont', 'Run or continue the ALMD calculation', cont_command),
        ('gen', 'Extract DFT results, generate training data, and execute MLIP training', gen_command),
        ('aiMD-rand', 'Run random aiMD sampling workflow', aiMD_rand_command),
        ('test', 'Check validation error', test_command),
        ('runMD', 'Run MD with trained models', runmd_command),
        ('cnvg', 'Check uncertainty convergence', cnvg_command),
    ]
    for name, help_text, func in commands:
        parser = subparsers.add_parser(name, help=help_text)
        parser.set_defaults(func=func)


def _add_utils_commands(subparsers):
    utils_parser = subparsers.add_parser('utils', help='Run utility commands')
    utils_subparsers = utils_parser.add_subparsers(dest='utility')
    utils_subparsers.required = True

    parser = utils_subparsers.add_parser(
        'aims2son',
        help='Convert aims.out to trajectory.son',
    )
    parser.add_argument('temperature', type=float, help='Temperature (K)')
    parser.set_defaults(func=aims2son_command)

    parser = utils_subparsers.add_parser(
        'split',
        help='Split trajectory.son into trajectory_test.son and trajectory_train.son',
    )
    parser.add_argument('num_split', type=int, help='The number of testing data')
    parser.add_argument('E_gs', type=float, help='Reference total energy in eV/unit cell')
    parser.add_argument('--harmonic_F', action='store_true', help='Use harmonic force constants')
    parser.set_defaults(func=split_son_command)

    parser = utils_subparsers.add_parser(
        'harmonic_run',
        help='Prepare DFT calculations from harmonic sampling structures',
    )
    parser.add_argument('temperature', type=float, help='Temperature (K)')
    parser.add_argument('num_sample', type=int, help='The number of harmonic samples')
    parser.add_argument('DFT_calc', type=str, help='The DFT calculator name')
    parser.add_argument('num_calc', type=int, help='The number of job scripts to write')
    parser.set_defaults(func=harmonic_run_command)

    parser = utils_subparsers.add_parser(
        'harmonic2son',
        help='Collect harmonic-sampling DFT results and convert them to SON',
    )
    parser.add_argument('temperature', type=float, help='Temperature (K)')
    parser.add_argument('num_sample', type=int, help='The number of harmonic samples')
    parser.add_argument('output_format', type=str, help='The output format to read')
    parser.set_defaults(func=harmonic2son_command)

    parser = utils_subparsers.add_parser(
        'traj_run',
        help='Prepare DFT calculations from a trajectory file',
    )
    parser.add_argument('traj_path', type=str, help='Path to the trajectory file')
    parser.add_argument('thermal_cutof''Thermalization cutoff')
    parser.add_argument('num_traj', type=int, help='The number of configurations for DFT')
    parser.add_argument('DFT_calc', type=str, help='The DFT calculator name')
    parser.add_argument('num_calc', type=int, help='The number of job scripts to write')
    parser.add_argument('--harmonic_F', action='store_true', help='Use harmonic force constants')
    parser.set_defaults(func=traj_run_command)

    parser = utils_subparsers.add_parser(
        'cnvg_post',
        help='Plot uncertainty convergence',
    )
    parser.add_argument('nmodel', type=int, help='The number of model ensemble members')
    parser.add_argument('nstep', type=int, help='The number of subsampling sets')
    parser.set_defaults(func=cnvg_post_command)

    parser = utils_subparsers.add_parser(
        'convert_npz',
        help='Include/exclude harmonic terms in/from an NPZ file',
    )
    parser.add_argument('name', type=str, help='The NPZ file name')
    parser.add_argument('--harmonic_F', action='store_true', help='Exclude harmonic terms')
    parser.set_defaults(func=convert_npz_command)


def build_parser():
    parser = argparse.ArgumentParser(prog='almomd')
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    _add_root_commands(subparsers)
    _add_utils_commands(subparsers)
    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
