"""Average the accuracy-rejection curves over seeds and write the csv files used by the ARC plots of the paper.

Reads  <results-dir>/accaccs/<dataset>_<seed>_<model>.dat
Writes <results-dir>/latex_plots/ARCs/csvs/<dataset>_<model>_arc.csv  (compiled by latex_plots/ARCs/stalone_ARC.tex)
"""
import argparse
from pathlib import Path

import numpy as np

from robustness_quantification.evaluation.store_results import read_accaccs

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = ['adult', 'Australian', 'bank', 'BCW', 'german', 'Healthy_aging', 'solar_big', 'solar_small',
            'spect_heart', 'student_math_bin', 'student_port_bin']
UNCERTAINTY_METRICS = ['max_prob', 'entropies', 'aleatoric', 'epistemic', 'total', 'margin_of_confidence']
ROBUSTNESS_METRICS = ['local_eps', 'total_var', 'chi2']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--datasets', nargs='+', default=DATASETS)
    p.add_argument('--models', nargs='+', choices=['NBC', 'GeFs'], default=['NBC', 'GeFs'])
    p.add_argument('--results-dir', type=Path, default=REPO_ROOT / 'results' / 'pgm26')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.results_dir / 'latex_plots' / 'ARCs' / 'csvs'
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = UNCERTAINTY_METRICS + ROBUSTNESS_METRICS

    for model in args.models:
        for dataset in args.datasets:
            per_seed = [read_accaccs(args.results_dir / 'accaccs' / f'{dataset}_{seed}_{model}.dat')
                        for seed in args.seeds]
            curves = {m: np.mean([[float(x) for x in curves[m]] for curves in per_seed], axis=0) for m in metrics}

            with open(out_dir / f'{dataset}_ALL_{model}_arc.csv', 'w') as f:
                f.write('n,' + ','.join(metrics) + '\n')
                for i in range(len(curves['local_eps'])):
                    f.write(str(i) + ''.join(',' + str(curves[m][i]) for m in metrics) + '\n')
    print(f'ARC csv files written to {out_dir}')


if __name__ == '__main__':
    main()
