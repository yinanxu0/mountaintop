# mountaintop
[![Python-Version](https://img.shields.io/badge/Python-3.7%7C3.8-brightgreen)](https://github.com/yinanxu0/mountaintop)

Easy to train a model. Will support more model in the future

## Install
```
pip3 install mountaintop
```

## Document
We afford python package and bin mode. For more details, please check `mountaintop -h`. 
```
usage: mountaintop [-h] [-v] {train,average,decode,wer,export,visual,eval} ...

MountainTop Line Interface

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show MountainTop version

subcommands:
  use "mountaintop [sub-command] --help" to get detailed information about each sub-command

  {train,average,decode,wer,export,visual,eval}
    train               👋 train a pytorch model
    average             👋 average a pytorch model
    decode              👋 decode a pytorch model
    wer                 👋 compute wer of reference and hypotheses
    export              👋 export a pytorch model
    visual              👋 visualize a pytorch model
    eval                👋 evaluate a pytorch model

mountaintop v0.1.1, a toolkit based on pytorch. Visit https://github.com/yinanxu0/mountaintop for tutorials and documents.
```
For convenience, you can use `mt` instead of `mountaintop`, like `mt -h`.

### Train model
```
torchrun --nnodes=${num_nodes} --nproc_per_node=${num_gpus} --no_python \
    mt train --config ${train_config} --model_dir ${model_dir} \
    --world_size $num_gpus --dist_backend $dist_backend \
    --num_workers 32 --tensorboard_dir ${tensorboard_dir} --pin_memory
```
`torchrun` is recommended to easily use multi-gpu and multi-machine.

More details of parameters in help mode.
```
mt train -h
```


### Evaluate model
```
mt average --model_dir ${model_dir} --val_best --num ${avg_num} --dst_model ${avg_pt}
fi

mt decode --config ${train_config} --mode ${mode} --gpu 0 --dict ${dict} \
    --checkpoint ${avg_pt} --num_workers ${num_workers} --ctc_weight ${ctc_weight} \
    --result_file ${decode_result} \
    ${chunk_size:+--chunk_size $chunk_size} \
    ${num_left_chunks:+--num_left_chunks $num_left_chunks}

mt wer --ref_file data/test.jsonl --hyp_file ${decode_result} --result_file ${wer_result}

```
