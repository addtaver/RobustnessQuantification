"""Write the csv files for the ARC grid of the UAI paper (all metrics, for several training sizes and noise levels).

Reads  <results-dir>/accaccs/<dataset>_<seed>_<size>_noise_<noise>_<model>.dat
Writes <results-dir>/latex_plots/ARC_grid/files/<dataset>_<size>_noise_<noise>_<model>_arc.csv
       (compiled by latex_plots/ARC_grid/figuren_mean_ARC_<model>.tex)
"""
import argparse
from pathlib import Path

from robustness_quantification.evaluation.store_results import read_accaccs

REPO_ROOT = Path(__file__).resolve().parents[2]

METRICS = ['margin_of_confidence', 'max_prob', 'aleatoric', 'total', 'epistemic', 'entropies',
           'global_eps', 'local_eps']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', default='student_port_bin')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--sizes', nargs='+', default=['50', '100', '200'])
    p.add_argument('--noises', nargs='+', default=['0.0', '0.1', '0.2'])
    p.add_argument('--models', nargs='+', choices=['NBC', 'GeFs'], default=['NBC', 'GeFs'])
    p.add_argument('--results-dir', type=Path, default=REPO_ROOT / 'results' / 'uai')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.results_dir / 'latex_plots' / 'ARC_grid' / 'files'
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        for size in args.sizes:
            for noise in args.noises:
                curves = read_accaccs(args.results_dir / 'accaccs'
                                      / f'{args.dataset}_{args.seed}_{size}_noise_{noise}_{model}.dat')
                with open(out_dir / f'{args.dataset}_{size}_noise_{noise}_{model}_arc.csv', 'w') as f:
                    f.write('n,' + ','.join(METRICS) + '\n')
                    for i in range(len(curves['global_eps'])):
                        f.write(str(i) + ''.join(',' + curves[m][i] for m in METRICS) + '\n')
    print(f'ARC grid csv files written to {out_dir}')


if __name__ == '__main__':
    main()
