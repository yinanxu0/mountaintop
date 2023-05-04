import torch


from mountaintop.core.internal.distribute import (
    get_global_rank, 
    get_local_rank,
    get_world_size,
    is_distributed,
    init_distributed,
)
from mountaintop.core.internal.module import import_module
from mountaintop.runx.trainer import Trainer
from mountaintop.runx.logx import loggerx
from mountaintop.utils.yaml import load_yaml
from mountaintop.layers.optimizer import get_optimizer, WarmupLR
from mountaintop.bin.parser_base import (
    set_base_parser,
    add_arg_group
)


def set_train_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, 'trainer arguments')
    
    gp.add_argument(
        '--config',
        type=str,
        default='CSB.yaml',
        help='a yaml file configs models')
    
    gp.add_argument(
        '--model_dir', 
        required=True, 
        type=str,
        metavar='PATH',
        help='saved model dir and '
    )
        
    gp.add_argument(
        '--checkpoint',
        default=None,
        type=str,
        metavar='PATH',
        help='path to the checkpoint (default: none)',
    )

    gp.add_argument(
        '--dist_backend',
        default='nccl',
        choices=['nccl', 'gloo'],
        help='distributed backend', 
    )
    gp.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='num of subprocess workers for reading, 0 to disable multiprocessing', 
    )
    
    gp.add_argument(
        '--resume_optimizer',
        action='store_true',
        default=False,
        help='resume optimizer (and scheduler) from checkpoint',
    )

    gp.add_argument(
        '--no_cuda',
        action='store_true',
        default=False,
        help='disables CUDA training')

    gp.add_argument(
        '--use_amp',
        action='store_true',
        default=False,
        help='use apex for automatic mixed precision training',
    )

    gp.add_argument(
        '--use_sync_bn',
        action='store_true',
        default=False,
        help=
        'convert BatchNorm layer to SyncBatchNorm before wrapping Network with DDP',
    )

    gp.add_argument(
        '--prefetch',
        default=100,
        type=int,
        help='prefetch number'
    )

    gp.add_argument(
        '--use_prefetcher',
        action='store_true',
        default=False,
        help='use prefetcher to speed up data loader',
    )
    
    gp.add_argument(
        '--pin_memory',
        action='store_true',
        default=False,
        help='use prefetcher to speed up data loader',
    )

    gp.add_argument(
        '--log_gradient',
        action='store_true',
        default=False,
        help='logging the model gradients',
    )

    return parser


def run(args, unused_args):
    ######## preparation before everything
    configs = load_yaml(args.config)
    assert "model" in configs
    assert "data_fetcher" in configs
    assert "dataset" in configs
    assert "train" in configs
    model_config = configs["model"]
    fetcher_config = configs["data_fetcher"]
    dataset_config = configs["dataset"]
    train_config = configs["train"]
    
    # init loggerx
    loggerx.initialize(
        logdir=args.model_dir,
        tensorboard=True,
        global_rank=get_global_rank(),
        eager_flush=True,
        hparams=configs,
    )
    
    local_rank = get_local_rank()
    world_size = get_world_size()
    loggerx.info(f"Local rank is {local_rank}")
    
    # Set random seed
    random_seed = 777
    loggerx.info(f"Setting random see to {random_seed}")
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    #################### init model
    # Init model from configs
    use_cuda = torch.cuda.is_available()
    assert "module" in model_config
    model_module_path = model_config.pop("module")
    loggerx.info(f'Init model from {model_module_path}')
    model_module = import_module(model_module_path)
    assert hasattr(model_module, "create_from_config"), \
        f"{model_module} should have init function [create_from_config]"
    model = model_module.create_from_config(model_config)
    loggerx.summary_model(model)

    ######## preparation before training
    if loggerx.is_master:
        # # !!!IMPORTANT!!!: Try to export the model by script, if fails, we 
        # # should refine the code to satisfy the script export requirements
        # TODO: 
        # script_model = torch.jit.script(model)
        # script_model.save(os.path.join(args.model_dir, 'init.zip'))
        pass

    use_dist = is_distributed()
    if use_dist:
        assert use_cuda
        # if args.fp16_grad_sync:
        #     from torch.distributed.algorithms.ddp_comm_hooks import (
        #         default as comm_hooks,
        #     )
        #     model.register_comm_hook(
        #         state=None, hook=comm_hooks.fp16_compress_hook
        #     )
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        init_distributed(
            use_horovod=False,
            backend=args.dist_backend,
            init_method=None, 
        )
        # dist.init_process_group(args.dist_backend)
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(
            model, find_unused_parameters=True, 
            device_ids=[local_rank], output_device=local_rank) 
    else:
        device = torch.device('cuda' if use_cuda else 'cpu')
        model = model.to(device)
    
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
    
    train_set_loader = data_loader_cls.create_data_loader(
        data_path=dataset_config["train"], 
        mode='train', 
    )
    valid_set_loader = data_loader_cls.create_data_loader(
        data_path=dataset_config["valid"], 
        mode='valid', 
    )
    
    ################## init trainer 
    optimizer_config = train_config.pop("optimizer", {})
    warmup_config = train_config.pop("warmup", {})
    optimizer = get_optimizer(model.parameters(), **optimizer_config)
    scheduler = WarmupLR(optimizer, **warmup_config)
    
    train_config['use_amp'] = args.use_amp
    num_epochs = train_config.pop('num_epochs', 100)
    
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        **train_config,
    )
    
    # If specify checkpoint, restore from checkpoint
    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint, resume=args.resume_optimizer)
    
    ################## training 
    trainer.run(
        train_loader=train_set_loader,
        eval_loaders=valid_set_loader,
        num_epochs=num_epochs,
    )
