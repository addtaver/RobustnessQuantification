"""Write the AUC table (all uncertainty metrics vs. global/local robustness and their hybrids) of the comparing_rq_and_uq paper.

Reads  <results-dir>/auc/<dataset>_<seed>_<size>_noise_<noise>_<model>.dat
Writes <results-dir>/latex_plots/AUCs/files/table_all_metrics_<seed>_<model>[_DS].csv
       (compiled by latex_plots/AUCs/standalone_table_<model>[_DS].tex; '_DS' = with label noise)
"""
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = ['Australian', 'adult', 'bank', 'BCW', 'german', 'Healthy_aging', 'heart_disease', 'lymphography',
            'nursery', 'solar_big', 'solar_small', 'spect_heart', 'student_math_bin', 'student_port_bin']
UNCERTAINTY_SHORT = ['margi', 'max_p', 'aleat', 'total', 'epist', 'entro']
PRECISION = 5


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--datasets', nargs='+', default=DATASETS)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--size', default='50')
    p.add_argument('--noise', type=float, default=0.0)
    p.add_argument('--models', nargs='+', choices=['NBC', 'GeFs'], default=['NBC', 'GeFs'])
    p.add_argument('--results-dir', type=Path, default=REPO_ROOT / 'results' / 'comparing_rq_and_uq')
    return p.parse_args(argv)


def parse_auc_file(path):
    """Extract the AUC of every metric from an auc file written by run_experiments.py."""
    with open(path, 'r') as f:
        lines = f.readlines()

    n_unc = len(UNCERTAINTY_SHORT)
    aucs = {
        'global': float(lines[0].split(':')[1].strip()[:PRECISION + 2]),
        'local': float(lines[2 * n_unc + 3].split(':')[1].strip()[:PRECISION + 2]),
    }
    for i, name in enumerate(UNCERTAINTY_SHORT):
        aucs[name] = float(lines[i * 2].split(':')[1].strip().split(' ')[1][:PRECISION + 2])
    return aucs


def main(argv=None):
    args = parse_args(argv)
    out_dir = args.results_dir / 'latex_plots' / 'AUCs' / 'files'
    out_dir.mkdir(parents=True, exist_ok=True)
    # column order expected (positionally) by the LaTeX tables
    order = ['max_p', 'margi', 'entro', 'total', 'aleat', 'epist', 'global', 'local']
    header = ['dataset', 'maxprob', 'margin', 'entropies', 'total', 'aleatoric', 'epistemic', 'robglobal', 'roblocal']
    for model in args.models:
        rows = []
        for i, dataset in enumerate(args.datasets):
            path = args.results_dir / 'auc' / f'{dataset}_{args.seed}_{args.size}_noise_{args.noise}_{model}.dat'
            aucs = parse_auc_file(path)
            rows.append([f'D{i + 1}'] + [f'{aucs[k]:.4f}' for k in order])

        suffix = '_DS' if args.noise > 0 else ''
        out_file = out_dir / f'table_all_metrics_{args.seed}_{model}{suffix}.csv'
        with open(out_file, 'w') as f:
            f.write(','.join(header) + '\n')
            for row in rows:
                f.write(','.join(row) + '\n')
        print(f'Table written to {out_file}')


if __name__ == '__main__':
    main()
