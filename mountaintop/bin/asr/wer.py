

import os
import sys
import codecs

from uphill import DocumentArray


from mountaintop.core.audio.wer import (
    WerCalculator,
    characterize,
    normalize,
    width,
    default_cluster,
)
from mountaintop.bin.parser_base import (
    set_base_parser,
    add_arg_group
)
from mountaintop import loggerx


def set_wer_parser(parser=None):
    if not parser:
        parser = set_base_parser()

    gp = add_arg_group(parser, 'wer arguments')
    
    gp.add_argument(
        '--ref_file', 
        required=True, 
        help='reference file'
    )
    
    gp.add_argument(
        '--hyp_file', 
        required=True, 
        help='hypotheses file'
    )
    
    gp.add_argument(
        '--result_file', 
        help='result file'
    )
    
    ### wer args
    gp.add_argument(
        '--max_words', 
        type=int,
        default=sys.maxsize,
        help='max words per line'
    )
    
    gp.add_argument(
        '--remove_tag', 
        action='store_true',
        default=True,
        help='remove tag'
    )
    
    gp.add_argument(
        '--case_sensitive', 
        action='store_true',
        default=False,
        help='case sensitive'
    )
    
    gp.add_argument(
        '--cluster_file', 
        default='',
        help='cluster file'
    )
    
    gp.add_argument(
        '--split_file', 
        default='',
        help='split file'
    )
    
    gp.add_argument(
        '--ignore_words', 
        default=[],
        help='ignore words'
    )
    
    gp.add_argument(
        '--tochar', 
        action='store_true',
        default=True,
        help='tochar'
    )
    
    gp.add_argument(
        '--padding_symbol', 
        choices=[
            'space', 'underline',
        ],
        default='space',
        help='padding symbol'
    )

    return parser


def run(args, unused_args):
    loggerx.initialize(to_file=False)
    calculator = WerCalculator()
    
    ignore_words = set(args.ignore_words)
    if not args.case_sensitive:
        ignore_words = set([w.lower() for w in args.ignore_words])

    language_clusters = {}
    seen_words = {}

    split = {}
    if os.path.exists(args.split_file):
        with codecs.open(args.split_file, 'r', 'utf-8') as fp:
            for line in fp:  # line in unicode
                words = line.strip().split()
                if len(words) >= 2:
                    split[words[0]] = words[1:]

    if split and not args.case_sensitive:
        newsplit = dict()
        for w in split:
            words = split[w]
            for i in range(len(words)):
                words[i] = words[i].lower()
            newsplit[w.lower()] = words
        split = newsplit

    padding_symbol = ' ' if args.padding_symbol == 'space' else '_'
    
    hyp_array = DocumentArray.from_file(args.hyp_file)
    ref_array = DocumentArray.from_file(args.ref_file)
    
    # compute error rate on the interaction of reference file and hyp file
    result_lines = []
    for key in ref_array.ids:
        if key not in hyp_array:
            continue
        
        line = f"utt: {key}"
        result_lines.append(line + "\n")
        
        hyp = hyp_array.load_text(key)
        hyp = characterize(hyp) if args.tochar else hyp
        hyp = normalize(hyp, ignore_words, args.case_sensitive, split, args.remove_tag)

        ref = ref_array.load_text(key)
        ref = characterize(ref) if args.tochar else ref
        ref = normalize(ref, ignore_words, args.case_sensitive, split, args.remove_tag)
        
        for word in hyp + ref :
            if word not in seen_words :
                language_name = default_cluster(word)
                if language_name not in language_clusters :
                    language_clusters[language_name] = {}
                if word not in language_clusters[language_name] :
                    language_clusters[language_name][word] = 1
                seen_words[word] = language_name

        result = calculator.calculate(ref, hyp)
                
        wer = 0.0
        if result['all'] != 0 :
            wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
        line = f"WER: {wer:.2f} % N={result['all']:d} C={result['cor']:d} "\
                f"S={result['sub']:d} D={result['del']:d} I={result['ins']:d}"
        result_lines.append(line + "\n")
        
        space = {}
        space['ref'] = []
        space['hyp'] = []
        for idx in range(len(result['ref'])) :
            len_lab = width(result['ref'][idx])
            len_rec = width(result['hyp'][idx])
            length = max(len_lab, len_rec)
            space['ref'].append(length-len_lab)
            space['hyp'].append(length-len_rec)
        upper_lab = len(result['ref'])
        upper_rec = len(result['hyp'])
        lab1, rec1 = 0, 0
        while lab1 < upper_lab or rec1 < upper_rec:
            line = "ref: "
            lab2 = min(upper_lab, lab1 + args.max_words)
            for idx in range(lab1, lab2):
                token = result['ref'][idx]
                line += f"{token}"
                for n in range(space['ref'][idx]) :
                    line += f"{padding_symbol}"
                line += ' '
            result_lines.append(line + "\n")
            line = "hyp: "
            rec2 = min(upper_rec, rec1 + args.max_words)
            for idx in range(rec1, rec2):
                token = result['hyp'][idx]
                line += f"{token}"
                for n in range(space['hyp'][idx]) :
                    line += f"{padding_symbol}"
                line += ' '
            result_lines.append(line + "\n\n")
            lab1 = lab2
            rec1 = rec2

    line = "="*80
    result_lines.append(line + "\n\n")
    
    result = calculator.overall()
    wer = 0.0
    if result['all'] != 0 :
        wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
    
    line = f"Overall -> {wer:.2f} % N={result['all']:d} C={result['cor']:d} "\
            f"S={result['sub']:d} D={result['del']:d} I={result['ins']:d}"
    result_lines.append(line + "\n")
    loggerx.info(line)

    for cluster_id in language_clusters :
        result = calculator.cluster([ k for k in language_clusters[cluster_id] ])
        wer = 0.0
        if result['all'] != 0 :
            wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
        
        line = f"{cluster_id} -> {wer:.2f} % N={result['all']:d} C={result['cor']:d} "\
                f"S={result['sub']:d} D={result['del']:d} I={result['ins']:d}"
        result_lines.append(line + "\n")
    result_lines.append("\n")
    if len(args.cluster_file) > 0 : # compute separated WERs for word clusters
        cluster_id = ''
        cluster = []
        for line in open(args.cluster_file, 'r', encoding='utf-8') :
            for token in line.decode('utf-8').rstrip('\n').split() :
                # end of cluster reached, like </Keyword>
                if token[0:2] == '</' and token[len(token)-1] == '>' and \
                    token.lstrip('</').rstrip('>') == cluster_id :
                    result = calculator.cluster(cluster)
                    wer = 0.0
                    if result['all'] != 0 :
                        wer = float(result['ins'] + result['sub'] + result['del']) * 100.0 / result['all']
                    line = f"{cluster_id} -> {wer:.2f} % N={result['all']:d} C={result['cor']:d} "\
                            f"S={result['sub']:d} D={result['del']:d} I={result['ins']:d}"
                    result_lines.append(line + "\n")
                    cluster_id = ''
                    cluster = []
                # begin of cluster reached, like <Keyword>
                elif token[0] == '<' and token[len(token)-1] == '>' and \
                        cluster_id == '' :
                    cluster_id = token.lstrip('<').rstrip('>')
                    cluster = []
                # general terms, like WEATHER / CAR / ...
                else :
                    cluster.append(token)
        result_lines.append("\n")
    line = "="*80
    result_lines.append(line + "\n")

    if args.result_file is not None:
        with open(args.result_file, "w") as fh:
            fh.writelines(result_lines)
    
