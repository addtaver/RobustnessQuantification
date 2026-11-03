"""Create the tables with the mean accuracy and the number of wins per rejection rate (+ the AU-ARC row).

Reads  <results-dir>/accaccs/<dataset>_<seed>_<model>.dat and <results-dir>/auc_summary/<seed>_<model>.dat
Writes <results-dir>/latex_plots/RR_ARC/csvs/table_rr_combi_<model>.csv  (compiled by latex_plots/RR_ARC/*.tex)
"""
import argparse
from pathlib import Path

import numpy as np

from robustness_quantification.evaluation.store_results import read_accaccs

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = ['adult', 'Australian', 'bank', 'BCW', 'german', 'Healthy_aging', 'solar_big', 'solar_small',
            'spect_heart', 'student_math_bin', 'student_port_bin']
REJECTION_RATES = [0.1, 0.25, 0.75, 0.9]
UNCERTAINTY_METRICS = ['total', 'aleatoric', 'epistemic', 'max_prob', 'entropies', 'margin_of_confidence']
ROBUSTNESS_METRICS = ['chi2', 'total_var', 'local_eps']
METRICS = ROBUSTNESS_METRICS + UNCERTAINTY_METRICS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--datasets', nargs='+', default=DATASETS)
    p.add_argument('--models', nargs='+', choices=['NBC', 'GeFs'], default=['NBC', 'GeFs'])
    p.add_argument('--results-dir', type=Path, default=REPO_ROOT / 'results' / 'pgm26')
    return p.parse_args(argv)


def read_aucs(auc_summary_file):
    """Parse an auc_summary file into {metric name as written in the file (max. 8 chars): [AUC per dataset]}."""
    aucs = {}
    with open(auc_summary_file, 'r') as f:
        for line in f.readlines()[2:]:
            if line == '\n':
                continue
            for entry in line.split(':')[1].strip().split('|'):
                name = entry.split('(')[0].strip()
                aucs.setdefault(name, []).append(float(entry.split('(')[1].split('±')[0].strip()))
    return aucs


def make_table(args, model):
    metrics = np.array(METRICS)
    mean_at_rr = [[] for _ in REJECTION_RATES]
    winners_at_rr = [[] for _ in REJECTION_RATES]

    for dataset in args.datasets:
        per_seed = [read_accaccs(args.results_dir / 'accaccs' / f'{dataset}_{seed}_{model}.dat')
                    for seed in args.seeds]
        curves = {m: np.mean([[float(x) for x in curves[m]] for curves in per_seed], axis=0) for m in METRICS}

        N = len(curves['local_eps'])
        for i, rr in enumerate(REJECTION_RATES):
            values = np.array([curves[m][int(N * rr)] for m in METRICS])
            winners_at_rr[i] += list(metrics[np.where(values == np.max(values))[0]])
            mean_at_rr[i].append(values)

    mean_at_rr = np.mean(np.array(mean_at_rr), axis=1)

    aucs = {}
    for seed in args.seeds:
        for name, values in read_aucs(args.results_dir / 'auc_summary' / f'{seed}_{model}.dat').items():
            aucs.setdefault(name, []).extend(values)

    out_dir = args.results_dir / 'latex_plots' / 'RR_ARC' / 'csvs'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f'table_rr_combi_{model}.csv', 'w') as f:
        f.write('dataset,' + ',,'.join(metrics) + ',\n')
        f.write(',' + ','.join(['\\%', 'W'] * len(metrics)) + '\n')

        for i, rr in enumerate(REJECTION_RATES):
            line = f'{rr * 100:.0f}\\%'
            for j, metric in enumerate(metrics):
                line += f',{mean_at_rr[i][j] * 100:.2f},{winners_at_rr[i].count(metric)}'
            f.write(line + '\n')

        line = '{\\footnotesize AU-ARC}'
        for metric in METRICS:
            line += f',\\SetCell[c=2]{{c}} {np.mean(aucs[metric[:8]]):.5f},'
        f.write(line + '\n')
    print(f'Table written to {out_dir / f"table_rr_combi_{model}.csv"}')


def main(argv=None):
    args = parse_args(argv)
    for model in args.models:
        make_table(args, model)


if __name__ == '__main__':
    main()
