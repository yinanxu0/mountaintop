
from mountaintop.bin.parser_base import (
    set_base_parser,
    add_arg_group
)
from mountaintop.bin.asr.decode import set_decode_parser
from mountaintop.bin.asr.wer import set_wer_parser


def set_asr_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, 'asr related commands')
    

    sp = parser.add_subparsers(
        dest='sub_cli',
        description='use "%(prog)-8s [sub-command] --help" '
        'to get detailed information about each sub-command',
    )
    
    # commands
    set_decode_parser(
        sp.add_parser(
            'decode',
            help='👋 decode a pytorch model',
            description='Start to decode a pytorch model, '
            'without any extra codes.',
        )
    )

    set_wer_parser(
        sp.add_parser(
            'wer',
            help='👋 compute wer of reference and hypotheses',
            description='Compute wer of reference and hypotheses '
            'without any extra codes.',
        )
    )

    return parser

