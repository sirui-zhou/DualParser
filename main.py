#!/usr/bin/env python3
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from dualparser import LogParser                                  # noqa: E402
from dualparser.evaluation.settings import benchmark_settings      # noqa: E402


def build_parser(args):
    setting = benchmark_settings[args.dataset]
    log_file = setting['log_file']
    data_type = args.data_type
    if data_type != '2k':
        log_file = log_file.replace('_2k', '_' + data_type)
    indir = os.path.join(args.data_dir, os.path.dirname(log_file))
    log_name = os.path.basename(log_file)
    if not os.path.exists(os.path.join(indir, log_name)):
        raise SystemExit('log file not found: %s' % os.path.join(indir, log_name))
    parser = LogParser(
        log_format=setting['log_format'],
        indir=indir,
        outdir=args.out_dir,
        rex=[],
        data_type=data_type,
        shot=args.shot,
        example_size=0,
        model=args.model,
        use_t5=args.use_t5,
        cold_start=args.cold_start,
        tpl_canon=not args.no_tpl_canon,
    )
    return parser, log_name


def evaluate(args):
    sys.path.insert(0, os.path.join(ROOT, 'dualparser'))
    import pandas as pd
    from evaluation.utils.evaluator_main import evaluate as _ga
    from evaluation.utils.PA_calculator import calculate_parsing_accuracy
    from evaluation.utils.template_level_analysis import evaluate_template_level

    setting = benchmark_settings[args.dataset]
    log_file = setting['log_file'].replace('_2k', '_' + args.data_type)
    gt_path = os.path.join(args.data_dir, log_file + '_structured.csv')
    pred_path = os.path.join(args.out_dir, os.path.basename(log_file) + '_structured.csv')
    gt = pd.read_csv(gt_path, dtype=str, keep_default_na=False)
    pred = pd.read_csv(pred_path, dtype=str, keep_default_na=False)
    if len(gt) != len(pred):
        raise SystemExit('row count mismatch: %d vs %d' % (len(gt), len(pred)))
    g = pd.DataFrame({'Content': gt['Content'], 'EventTemplate': gt['EventTemplate']})
    p = pd.DataFrame({'Content': gt['Content'], 'EventTemplate': pred['EventTemplate']})
    ga, fga = _ga(g, p)
    pa = calculate_parsing_accuracy(g, p)
    _, _, fta, pta, rta = evaluate_template_level(args.dataset, g, p)
    print('\n=== %s ===' % args.dataset)
    print('  GA  = %.2f%%' % (100 * ga))
    print('  PA  = %.2f%%' % (100 * pa))
    print('  FGA = %.2f%%' % (100 * fga))
    print('  FTA = %.2f%%' % (100 * fta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='dataset name, e.g. HDFS')
    ap.add_argument('--data_dir', default='datasets',
                    help='directory containing <dataset>/<dataset>_full.log')
    ap.add_argument('--out_dir', default='result')
    ap.add_argument('--data_type', default='full', choices=['2k', 'full', 'sub20000'])
    ap.add_argument('--model', default='deepseek-chat')
    ap.add_argument('--shot', type=int, default=0)
    ap.add_argument('--use_t5', action='store_true',
                    help='enable the optional T5 second-pass check (off by default)')
    ap.add_argument('--cold_start', action='store_true',
                    help='ignore any existing cache and parse from scratch')
    ap.add_argument('--no_tpl_canon', action='store_true')
    ap.add_argument('--eval', action='store_true', help='compute GA/PA/FGA/FTA after parsing')
    args = ap.parse_args()

    parser, log_name = build_parser(args)
    parser.parse(log_name)
    if args.eval:
        evaluate(args)


if __name__ == '__main__':
    main()
