"""Reproduce all results of the PGM'26 paper
"Local Robustness Quantification for Naive Bayes Classifiers and Generative Forests: a General Approach".

Steps (each one is a script in experiments/pgm26/):
    1. run_experiments.py   run NBC and GeF on all datasets for the given seeds   -> results/pgm26/{csv,accaccs,auc_summary}
    2. make_arc_csvs.py     accuracy-rejection curves for the ARC figures         -> results/pgm26/latex_plots/ARCs/csvs
    3. make_rr_tables.py    mean accuracy / wins per rejection rate + AU-ARC      -> results/pgm26/latex_plots/RR_ARC/csvs

The figures/tables themselves are then compiled from the .tex files in results/pgm26/latex_plots (lualatex).

Usage:
    python script/reproduce_PGM_paper.py            # full reproduction (takes a long time)
    python script/reproduce_PGM_paper.py --quick    # small smoke test on 3 datasets, written to results/pgm26_quick
    python script/reproduce_PGM_paper.py --skip-run # only rebuild the tables/figure data from existing raw results
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / 'experiments' / 'pgm26'

QUICK_DATASETS = ['BCW', 'spect_heart', 'Australian']


def run(script, *args):
    cmd = [sys.executable, str(EXPERIMENTS / script), *map(str, args)]
    print('\n$', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--quick', action='store_true', help='tiny configuration to check that the pipeline works')
    p.add_argument('--skip-run', action='store_true', help='skip the (slow) experiments, only post-process results')
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--results-dir', type=Path, default=None)
    args = p.parse_args()

    if args.quick:
        results_dir = args.results_dir or REPO_ROOT / 'results' / 'pgm26_quick'
        common = ['--seeds', *args.seeds[:1], '--datasets', *QUICK_DATASETS, '--results-dir', results_dir]
        run_args = [*common, '--n-runs', 2, '--n-estimators', 8]
    else:
        results_dir = args.results_dir or REPO_ROOT / 'results' / 'pgm26'
        common = ['--seeds', *args.seeds, '--results-dir', results_dir]
        run_args = common

    if not args.skip_run:
        run('run_experiments.py', *run_args)
    run('make_arc_csvs.py', *common)
    run('make_rr_tables.py', *common)
    print(f'\nDone. Results are in {results_dir}')


if __name__ == '__main__':
    main()
