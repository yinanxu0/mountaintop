from collections import Counter
import os
import torch
import glob


from mountaintop.bin.parser_base import (
    set_base_parser,
    add_arg_group
)
from mountaintop.runx.mixins.saver_mixins import SaverMixins
from mountaintop.runx.logx import loggerx


def set_average_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, "average model")

    gp.add_argument("--model_dir", "-m",
                    required=True,
                    help="model directory path to be averaged")
    gp.add_argument("--epoch", "-e",
                    default=0,
                    type=int,
                    help="last epoch used for averaging model")
    gp.add_argument("--by_metric",
                    action='store_true',
                    default=False,
                    help="average model by metric")
    
    gp.add_argument("--avg", "-a",
                    default=0,
                    type=int,
                    help="num of models used for averaging model")
    
    gp.add_argument("--higher_better",
                    action='store_true',
                    default=False,
                    help="check model metric")
    return parser
    

def load_metrics(folder, num, higher_better=False):
    filename_pattern = "ckpt_ep[0-9]*.pth"
    filepaths = glob.glob(os.path.join(folder, filename_pattern))
    metrics_meta = {}
    for filepath in filepaths:
        _, meta = loggerx.load_model(filepath)
        metric = meta["__metric"] if higher_better else -meta["__metric"]
        metrics_meta[filepath] = metric
    keep_filepaths = []
    for filepath, metric in Counter(metrics_meta).most_common(num):
        keep_filepaths.append(filepath)
    return keep_filepaths


def run(args, unused_args):
    # loggerx = get_console_logger()
    # init loggerx
    loggerx.initialize(
        logdir=args.model_dir,
        tensorboard=False,
        to_file=False
    )
    
    if args.epoch > 0 and args.by_metric:
        msg = "Wrong Command, average only handle one arg between --epoch and --by_metric"
        loggerx.critical(msg)
        return
    
    if args.epoch > 0:
        if args.avg > 0:
            start = args.epoch - args.avg
            assert start > 0, f"difference between epoch and avg should be positive not {start}"
            filepath_start = f"{args.model_dir}/ckpt_ep{start:04d}.pth"
            filepath_end = f"{args.model_dir}/ckpt_ep{args.epoch:04d}.pth"
            loggerx.info(
                f"Calculating the averaged model over epoch range in ({start}, {args.epoch}]"
            )
            avg_model = SaverMixins().load_average_checkpoints(
                filepath_start=filepath_start,
                filepath_end=filepath_end,
                device="cpu",
            )
            avg_filename = f"{args.model_dir}/avg_ckpt_ep{args.epoch:04d}_avg{args.avg}.pth"
            loggerx.info(f'Saving averaged model to {avg_filename}')
            torch.save(avg_model, avg_filename)
        else:
            filepath = f"{args.model_dir}/ckpt_ep{args.epoch:04d}.pth"
            checkpoint = SaverMixins().load_checkpoint(filepath=filepath, device="cpu")
            avg_model = checkpoint["__model"]
            avg_filename = f"{args.model_dir}/avg_ckpt_ep{args.epoch:04d}_avg{args.avg}.pth"
            loggerx.info(f'Saving averaged model to {avg_filename}')
            torch.save(avg_model, avg_filename)
    
    elif args.by_metric:
        assert args.avg > 0, "avg should > 0"
        keep_filepaths = load_metrics(args.model_dir, args.avg, args.higher_better)
        assert args.avg >= len(keep_filepaths)
        avg = min(args.avg, len(keep_filepaths))
        
        avg_model = None
        for filepath in keep_filepaths:
            model_dict, meta = loggerx.load_model(filepath)
            
            if avg_model is None:
                avg_model = model_dict
            else:
                for key in avg_model.keys():
                    # avg_model[key] += torch.true_divide(model_dict[key], avg)
                    
                    avg_model[key] += model_dict[key]
        # average
        for k in avg_model.keys():
            if avg_model[k] is not None:
                avg_model[k] = torch.true_divide(avg_model[k], avg)
        
        avg_filename = f"{args.model_dir}/avg_ckpt_ep{args.epoch:04d}_avg{avg}.pth"
        loggerx.info(f'Saving averaged model to {avg_filename}')
        torch.save(avg_model, avg_filename)
    
    else:
        msg = "Wrong Command, usage references: "\
                "\n\tcase1: mountaintop average -m [MODEL_DIR] -e [EPOCH] "\
                "\n\tcase2: mountaintop average -m [MODEL_DIR] -e [EPOCH] -a [AVG]"\
                "\n\tcase3: mountaintop average -m [MODEL_DIR] -a [AVG] --by_metric [--higher_better]"
        loggerx.critical(msg)
        return