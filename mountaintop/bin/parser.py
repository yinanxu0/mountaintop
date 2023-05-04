from mountaintop.bin.parser_base import set_base_parser
from mountaintop.bin.average import set_average_parser
from mountaintop.bin.train import set_train_parser
from mountaintop.bin.export import set_export_parser
from mountaintop.bin.visual import set_visualization_parser
from mountaintop.bin.eval import set_eval_parser
from mountaintop.bin.asr import set_asr_parser
from mountaintop.bin.profiler import set_profiler_parser

def get_main_parser():
    # create the top-level parser
    parser = set_base_parser()

    sp = parser.add_subparsers(
        dest='main_cli',
        description='use "%(prog)-8s [sub-command] --help" '
        'to get detailed information about each sub-command',
    )

    ## original parts
    set_train_parser(
        sp.add_parser(
            'train',
            help='👋 train a pytorch model',
            description='Start to train a pytorch model, '
            'without any extra codes.',
        )
    )
    
    set_average_parser(
        sp.add_parser(
            'average',
            help='👋 average a pytorch model',
            description='Start to average a pytorch model, '
            'without any extra codes.',
        )
    )

    set_export_parser(
        sp.add_parser(
            'export',
            help='🫗 export a pytorch model',
            description='Start to export a pytorch model, '
            'without any extra codes.',
        )
    )

    set_visualization_parser(
        sp.add_parser(
            'visual',
            help='👀 visualize a pytorch model',
            description='Start to visualize a pytorch model, '
            'without any extra codes.',
        )
    )
    
    set_eval_parser(
        sp.add_parser(
            'eval',
            help='💯 evaluate a pytorch model',
            description='Start to eval a pytorch model, '
            'without any extra codes.',
        )
    )
    
    set_profiler_parser(
        sp.add_parser(
            'profiler',
            help='⌚️ profiler a pytorch model',
            description='Start to profiler a pytorch model, '
            'without any extra codes.',
        )
    )
    
    ## asr part
    set_asr_parser(
        sp.add_parser(
            'asr',
            help='ASR part',
            description='Start to eval a pytorch model, '
            'without any extra codes.',
        )
    )

    return parser
