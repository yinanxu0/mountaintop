import os
import torch


from mountaintop.core.internal.distribute import (
    get_local_rank,
)
from mountaintop.core.internal.module import import_module
from mountaintop.runx.trainer import Trainer
from mountaintop.runx.logx import loggerx
from mountaintop.utils.yaml import load_yaml
from mountaintop.bin.parser_base import (
    set_base_parser,
    add_arg_group,
)


def set_profiler_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, 'profile arguments')

    gp.add_argument('--config', required=True, help='config file')
    
    gp.add_argument(
        '--out',
        type=str,
        required=True,
        default="profiler.txt",
        help='trace profile file store path'
    )
    
    gp.add_argument(
        '--format',
        choices=["chrome", "txt"],
        type=str,
        default="txt",
        help='trace profile file format'
    )
    
    gp.add_argument(
        '--no-cuda',
        action='store_true',
        default=False,
        help='disables CUDA training'
    )

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


def run(args, unused_args):
    ######## preparation before everything
    configs = load_yaml(args.config)
    assert "model" in configs
    assert "data_fetcher" in configs
    assert "dataset" in configs
    model_config = configs["model"]
    fetcher_config = configs["data_fetcher"]
    dataset_config = configs["dataset"]
    
    model_dir = ".tmp"
    # init magic logger
    loggerx.initialize(
        logdir=model_dir,
        to_file=False
    )
    local_rank = get_local_rank()
    loggerx.info(f"Local rank is {local_rank}")

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device(f'cuda:{local_rank}' if use_cuda else 'cpu')
    loggerx.info(f"Using device {device}")

    #################### init model
    # init model
    loggerx.info('Init model')
    assert "module" in model_config
    model_module_path = model_config.pop("module")
    model_module = import_module(model_module_path)
    assert hasattr(model_module, "create_from_config"), \
        f"{model_module} should have init function [create_from_config]"
    model = model_module.create_from_config(model_config).to(device)
    loggerx.summary_model(model)
    
    trainer = Trainer(
        model=model,
        optimizer=None,
    )
    
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
    
    def reorganize_data(data):
        if 'cuda' in str(device):
            for key, tensor in data.items():
                if isinstance(tensor, torch.Tensor):
                    data[key] = tensor.to(device, non_blocking=True)
        return data
        
    loggerx.info("Warmup model......")
    for batch_idx, batch_data in enumerate(test_set_loader):
        if batch_idx > 10:
            break
        trainer.iteration(reorganize_data(batch_data))
    loggerx.info("Warmup model finish :)")
    
    with torch.autograd.profiler.profile(
        enabled=True, use_cuda=use_cuda, record_shapes=True, profile_memory=True,
        with_stack=True
    ) as profiler:
        trainer.iteration(reorganize_data(batch_data))
    
    if args.format == "chrome":
        profiler.export_chrome_trace(args.out)
        loggerx.info(
            "profiler analysis finished. To trace the result, Please open " +
            f"Chrome brower, and input `chrome://tracing` with {args.out} " +
            "for details."
        )
    elif args.format == "txt":
        information = profiler.table()
        with open(args.out, "w") as fp:
            fp.write(information)
        loggerx.info(
            "profiler analysis finished. To trace the result, Please open " +
            f"{args.out} for details."
        )
            
