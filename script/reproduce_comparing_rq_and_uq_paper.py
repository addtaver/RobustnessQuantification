"""Reproduce the results of the comparing_rq_and_uq paper experiments (label-noise / limited-data setting).

Steps (each one is a script in experiments/comparing_rq_and_uq/):
    1. run_experiments.py   NBC and GeF, all metrics + hybrids       -> results/comparing_rq_and_uq/{csv,accaccs,auc,summaries}
    2. make_arc_grid.py     ARC grid data                             -> results/comparing_rq_and_uq/latex_plots/ARC_grid/files
    3. make_auc_tables.py   AUC tables                                -> results/comparing_rq_and_uq/latex_plots/AUCs/files

Usage:
    python script/reproduce_comparing_rq_and_uq_paper.py            # full reproduction (takes a long time)
    python script/reproduce_comparing_rq_and_uq_paper.py --quick    # small smoke test on BCW, written to results/comparing_rq_and_uq_quick
    python script/reproduce_comparing_rq_and_uq_paper.py --skip-run # only rebuild the tables/figure data from existing raw results

Note: the paper's tables use seed 42 and the arc grid seed 0 (see the defaults of the individual scripts);
pass --seed to run_experiments.py directly to generate other seeds.
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = REPO_ROOT / 'experiments' / 'comparing_rq_and_uq'


def run(script, *args):
    cmd = [sys.executable, str(EXPERIMENTS / script), *map(str, args)]
    print('\n$', ' '.join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--quick', action='store_true', help='tiny configuration to check that the pipeline works')
    p.add_argument('--skip-run', action='store_true', help='skip the (slow) experiments, only post-process results')
    p.add_argument('--results-dir', type=Path, default=None)
    args = p.parse_args()

    if args.quick:
        results_dir = args.results_dir or REPO_ROOT / 'results' / 'comparing_rq_and_uq_quick'
        if not args.skip_run:
            run('run_experiments.py', '--seed', 0, '--datasets', 'BCW', '--training-sizes', 50,
                '--noise-factors', 0.0, 0.2, '--n-runs', 2, '--n-estimators', 8, '--results-dir', results_dir)
        run('make_arc_grid.py', '--dataset', 'BCW', '--seed', 0, '--sizes', 50, '--noises', 0.0, 0.2,
            '--results-dir', results_dir)
        run('make_auc_tables.py', '--datasets', 'BCW', '--seed', 0, '--size', 50, '--results-dir', results_dir)
    else:
        results_dir = args.results_dir or REPO_ROOT / 'results' / 'comparing_rq_and_uq'
        if not args.skip_run:
            # the paper reports seed 0 (ARC grid) and seed 42 (tables)
            for seed in (0, 42):
                run('run_experiments.py', '--seed', seed, '--results-dir', results_dir)
        run('make_arc_grid.py', '--results-dir', results_dir)
        run('make_auc_tables.py', '--results-dir', results_dir)
        run('make_auc_tables.py', '--results-dir', results_dir, '--noise', 0.2)
    print(f'\nDone. Results are in {results_dir}')


if __name__ == '__main__':
    main()
