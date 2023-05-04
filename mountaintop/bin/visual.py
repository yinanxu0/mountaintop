import os
import torch
from torchviz import make_dot
from graphviz.backend.execute import ExecutableNotFound


from mountaintop.core.internal.distribute import (
    get_local_rank,
)
from mountaintop.core.internal.module import import_module
from mountaintop.runx.logx import loggerx
from mountaintop.utils.yaml import load_yaml
from mountaintop.bin.parser_base import (
    set_base_parser,
    add_arg_group
)


def set_visualization_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, 'visualization arguments')

    gp.add_argument(
        '--config', 
        required=True, 
        help='config file to initialize model'
    )

    gp.add_argument(
        '--save_path', 
        required=True, 
        help='file to save model visualization'
    )

    return parser


def run(args, unused_args):
    device = torch.device("cpu")
    
    configs = load_yaml(args.config)
    assert "model" in configs
    assert "data_fetcher" in configs
    assert "dataset" in configs
    model_config = configs["model"]
    fetcher_config = configs["data_fetcher"]
    dataset_config = configs["dataset"]
    
    # init magic logger
    loggerx.initialize(
        logdir="./.tmp",
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
    assert hasattr(model_module, "create_from_config"), \
        f"{model_module} should have init function [create_from_config]"
    model = model_module.create_from_config(model_config)
    loggerx.summary_model(model)
    
    ################## load dataset    
    assert "module" in fetcher_config
    data_loader_module_path = fetcher_config.pop("module")
    data_loader_module = import_module(data_loader_module_path)
    data_loader_cls = data_loader_module(
        fetcher_configs=fetcher_config,
        pin_memory=False, 
        num_workers=2, 
        prefetch_factor=2,
    )
    assert hasattr(data_loader_cls, "create_data_loader"), \
        f"{data_loader_module} should have function [create_data_loader]"
    data_loader = data_loader_cls.create_data_loader(
        data_path=dataset_config["valid"], 
        mode='valid', 
    )
    
    for batch_data in data_loader:
        assert isinstance(batch_data, dict), \
            f"batch_data in data_loader should be dictionary"
        # move batch of samples to device
        if 'cuda' in str(device):
            for key, tensor in batch_data.items():
                if isinstance(tensor, torch.Tensor):
                    batch_data[key] = tensor.to(device, non_blocking=True)
        output, loss, metrics = model(**batch_data)
        break
    net_graph = make_dot(loss, params=dict(list(model.named_parameters())))
    net_graph.directory = os.path.dirname(args.save_path)
    try:
        net_graph.render(os.path.basename(args.save_path), view=False)
        loggerx.info(f"save model visualization to {args.save_path}.pdf")
    except ExecutableNotFound as not_found_error:
        loggerx.error(not_found_error)
        loggerx.error("you maybe should install graphviz using: apt-get install graphviz")
    