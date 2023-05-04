import argparse
from termcolor import colored


from mountaintop import __version__


def add_arg_group(parser, title):
    parser = parser.add_argument_group(title)
    _add_common_args(parser)
    return parser


def set_base_parser():
    parser = argparse.ArgumentParser(
        epilog='%s, a toolkit based on pytorch. '
        'Visit %s for tutorials and documents.' % (
            colored('mountaintop v%s' % __version__, 'green'),
            colored(
                'https://github.com/yinanxu0/mountaintop',
                'cyan',
                attrs=['underline'],
            ),
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='MountainTop Line Interface',
    )

    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=__version__,
        help='show MountainTop version',
    )

    return parser


def _add_common_args(parser):
    parser.add_argument(
        '--dry_run',
        action='store_true',
        required=False,
        help='show input commands',
    )
