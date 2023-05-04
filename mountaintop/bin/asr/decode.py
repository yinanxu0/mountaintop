import os
import torch
from uphill import TextDocumentArray, Document
from uphill.core.text.vocab import Vocabulary


from mountaintop.core.internal.distribute import (
    get_global_rank, 
    get_local_rank,
    get_world_size,
)
from mountaintop.core.internal.module import import_module
from mountaintop.core.internal.timing import Timer
from mountaintop.runx.logx import loggerx
from mountaintop.utils.yaml import load_yaml
from mountaintop.models.saver import restore_model
from mountaintop.models.asr.utils import EncoderMode
from mountaintop.bin.parser_base import (
    set_base_parser,
    add_arg_group
)


def set_decode_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, 'decode arguments')
    
    gp.add_argument('--config', required=True, help='config file')
    
    gp.add_argument(
        '--gpu',
        type=int,
        default=-1,
        help='gpu id for this rank, -1 for cpu'
    )
    gp.add_argument(
        '--checkpoint', required=True, help='checkpoint model'
    )
    gp.add_argument(
        '--dict', 
        required=True, 
        help='vocabulary file path'
    )
    gp.add_argument('--non_lang_syms',
        help='non-linguistic symbol file. One symbol per line.'
    )
    gp.add_argument(
        '--beam_size',
        type=int,
        default=10,
        help='beam size for search'
    )
    gp.add_argument(
        '--penalty',
        type=float,
        default=0.0,
        help='length penalty'
    )
    gp.add_argument(
        '--result_file', 
        required=True, 
        help='asr result file'
    )
    gp.add_argument(
        '--mode',
        choices=['attention', 'ctc_greedy', 'ctc_beam', 'rescore'],
        default='attention',
        help='decoding mode'
    )
    gp.add_argument(
        '--context_mode',
        choices=['offline', 'dynamic_chunk', 'static_chunk', 'stream'],
        default='offline',
        help='encoding mode'
    )
    gp.add_argument(
        '--chunk_size',
        type=int,
        default=-1,
        help='''decoding chunk size,
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here'''
    )
    gp.add_argument(
        '--num_left_chunks',
        type=int,
        default=-1,
        help='number of left chunks for decoding'
    )
    gp.add_argument(
        '--ctc_weight',
        type=float,
        default=0.0,
        help='ctc weight for attention rescoring decode mode'
    )
    gp.add_argument(
        '--reverse_weight',
        type=float,
        default=0.0,
        help='''right to left weight for attention rescoring
                decode mode'''
    )
    gp.add_argument(
        '--bpe_model',
        default=None,
        type=str,
        help='bpe model for english part'
    )
    gp.add_argument(
        '--connect_symbol',
        default='',
        type=str,
        help='used to connect the output characters'
    )
    gp.add_argument(
        '--pin_memory',
        action='store_true',
        default=False,
        help='Use pinned memory buffers used for reading'
    )
    gp.add_argument(
        '--prefetch',
        default=100,
        type=int,
        help='prefetch number'
    )
    gp.add_argument(
        '--num_workers',
        default=4,
        type=int,
        help='num of subprocess workers for reading, 0 to disable multiprocessing'
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
        tensorboard=False,
        global_rank=get_global_rank(),
        to_file=False
    )
    local_rank = get_local_rank()
    world_size = get_world_size()
    loggerx.info(f"Local rank is {local_rank}")
    
    # if args.mode in ['ctc_beam', 'rescore'] and fetcher_config["batch"]["capcity"] > 1:
    #     logger.warning('decoding mode {} must be running with batch_size == 1'.format(args.mode))
    #     fetcher_config["batch"] = {
    #         "batch_type": "static",
    #         "capcity": 1
    #     }

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
    # restore model
    restore_model(model, args.checkpoint)
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    loggerx.info(f"using {device}")
    model = model.to(device)
    model.eval()
    
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
        mode='test', 
    )

    ################## Load vocab
    if "tokenizer" in fetcher_config and "vocab_path" in fetcher_config["tokenizer"]:
        vocab_path = fetcher_config["tokenizer"]["vocab_path"]
        if os.path.exists(vocab_path):
            loggerx.info(f"Using default vocab[{vocab_path}]")
        else:
            vocab_path = args.dict
            loggerx.info(f"Using input vocab[{vocab_path}]")
    vocab = Vocabulary(vocab_path=vocab_path)
    eos = len(vocab) - 1

    ################## inference
    timer = Timer()
    hyp_array = TextDocumentArray()
    with torch.no_grad():
        for batch_data in test_set_loader:
            # move batch of samples to device
            if 'cuda' in str(device):
                for key, tensor in batch_data.items():
                    if isinstance(tensor, torch.Tensor):
                        batch_data[key] = tensor.to(device, non_blocking=True)
            keys = batch_data["keys"]
            feat = batch_data["feat"]
            feat_lengths = batch_data["feat_lengths"]
            
            function_name = f"decode_by_{args.mode}"
            assert hasattr(model, function_name)
            
            context_mode = EncoderMode.to_enum(args.context_mode)
            assert context_mode != EncoderMode.NotValid
            decode_args = {
                "feat": feat,
                "feat_lengths": feat_lengths,
                "mode": args.context_mode,
                "beam_size": args.beam_size,
                "chunk_size": args.chunk_size,
                "num_left_chunks": args.num_left_chunks,
            }
            if args.mode == "rescore":
                decode_args["ctc_weight"] = args.ctc_weight
                decode_args["reverse_weight"] = args.reverse_weight
            
            results = getattr(model, function_name)(**decode_args)
            hyps = results[0] if args.mode == "rescore" else results

            for i, key in enumerate(keys):
                content = []
                for w in hyps[i]:
                    if w == eos:
                        break
                    content.append(vocab.id2token(w))
                text = args.connect_symbol.join(content)
                hyp_doc = Document.from_text(text=text, id=key)
                hyp_array.append(hyp_doc)
                if len(hyp_array) % 500 == 0:
                    loggerx.info(f"finish decoding {len(hyp_array)} examples, time elapsed: {timer.tick()}")
    hyp_array.to_file(args.result_file)
    loggerx.info(f"time elapsed: {timer.tock()}")
