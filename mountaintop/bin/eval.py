import os
from tabulate import tabulate


from mountaintop.core.internal.distribute import (
    get_local_rank,
)
from mountaintop.core.internal.module import import_module
from mountaintop.runx.trainer import Trainer
from mountaintop.runx.logx import loggerx
from mountaintop.utils.yaml import load_yaml
from mountaintop.bin.parser_base import (
    set_base_parser,
    add_arg_group
)


def set_eval_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, 'evaluator arguments')

    gp.add_argument('--config', required=True, help='config file')
    
    gp.add_argument(
        '--gpu',
        type=int,
        default=-1,
        help='gpu id for this rank, -1 for cpu'
    )
    gp.add_argument(
        '--checkpoint', 
        required=True, 
        type=str,
        metavar='PATH',
        help='path to the checkpoint'
    )

    gp.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')

    gp.add_argument(
        '--use-prefetcher',
        action='store_true',
        default=False,
        help='use prefetcher to speed up data loader',
    )
    
    gp.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='num of subprocess workers for reading, 0 to disable multiprocessing'
    )
    
    gp.add_argument(
        '--prefetch',
        default=100,
        type=int,
        help='prefetch number'
    )
    
    gp.add_argument(
        '--pin_memory',
        action='store_true',
        default=False,
        help='Use pinned memory buffers used for reading'
    )

    return parser


def remove_dropout(configs):
    for key, values in configs.items():
        if not isinstance(values, dict):
            continue
        for subkey, subvalue in values.items():
            if "dropout" in subkey:
                configs[key][subkey] = 0.0
    return


def run(args, unused_args):
    ######## preparation before everything
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    configs = load_yaml(args.config)
    assert "model" in configs
    assert "data_fetcher" in configs
    assert "dataset" in configs
    model_config = configs["model"]
    fetcher_config = configs["data_fetcher"]
    dataset_config = configs["dataset"]
    
    model_dir = os.path.dirname(args.checkpoint)
    # init magic logger
    loggerx.initialize(
        logdir=model_dir,
        # tensorboard=False,
        # global_rank=get_global_rank(),
        # eager_flush=False,
        # hparams=None,
        to_file=False
    )
    local_rank = get_local_rank()
    loggerx.info(f"Local rank is {local_rank}")

    #################### init model
    # init model
    loggerx.info('Init model')
    assert "module" in model_config
    model_module_path = model_config.pop("module")
    model_module = import_module(model_module_path)
    remove_dropout(model_config)
    assert hasattr(model_module, "create_from_config"), \
        f"{model_module} should have init function [create_from_config]"
    model = model_module.create_from_config(model_config)
    loggerx.summary_model(model)
    
    trainer = Trainer(
        model=model,
        optimizer=None,
    )
    
    # If specify checkpoint, restore from checkpoint
    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint, resume=False)
    
    
    ################## load dataset  
    assert "module" in fetcher_config
    data_loader_module_path = fetcher_config.pop("module")
    data_loader_module = import_module(data_loader_module_path)
    prefetch_factor = args.prefetch if args.num_workers > 0 else 2 # multiprocessing mode
    data_loader_cls = data_loader_module(
        fetcher_configs=fetcher_config,
        pin_memory=args.pin_memory, 
        num_workers=args.num_workers, 
        prefetch_factor=prefetch_factor,
    )
    assert hasattr(data_loader_cls, "create_data_loader"), \
        f"{data_loader_module} should have function [create_data_loader]"
    test_set_loader = data_loader_cls.create_data_loader(
        data_path=dataset_config["test"], 
        mode='valid', 
    )
    
    metrics = trainer.valid(test_set_loader)
    
    header = list(metrics.keys())
    table_data = [
        ["name"] + header,
        ["value"] + [metrics[h] for h in header]
    ]

    result = tabulate(table_data[1:], table_data[0], "pipe", floatfmt=".4f")
    loggerx.info(f"Evaluation result: \n{result}")

