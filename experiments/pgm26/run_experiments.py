"""Run the PGM'26 experiments: NBC and GeF reliability metrics (uncertainty + robustness) on the benchmark datasets.

For every seed and dataset, ``--n-runs`` random train/test splits are evaluated. Per dataset this writes, in
``<results-dir>``:
    csv/<dataset>_<seed>_<model>.csv     raw classification results and metric values per test instance
    accaccs/<dataset>_<seed>_<model>.dat accuracy-rejection curves (averaged over the runs), one per metric
    auc_summary/<seed>_<model>.dat       AUC of every metric (averaged over the runs), sorted per dataset
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from robustness_quantification.data.loading import load_dataset
from robustness_quantification.data.splits import split_train_test
from robustness_quantification.evaluation.accrej_plots import get_mean_accs
from robustness_quantification.evaluation.store_results import write_results_to_csv
from robustness_quantification.pipelines.gef import standard_experiment_GeFs
from robustness_quantification.pipelines.nbc import standard_experiment_NBC

REPO_ROOT = Path(__file__).resolve().parents[2]

PAPER_DATASETS = ['solar_small', 'Australian', 'student_math_bin', 'student_port_bin', 'german', 'Healthy_aging',
                  'solar_big', 'bank', 'adult', 'BCW', 'spect_heart']
UNCERTAINTY_METRICS = ['max_prob', 'entropies', 'aleatoric', 'epistemic', 'total', 'margin_of_confidence']
ROBUSTNESS_METRICS = ['local_eps', 'chi2', 'total_var']
METRICS = UNCERTAINTY_METRICS + ROBUSTNESS_METRICS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1])
    p.add_argument('--datasets', nargs='+', default=PAPER_DATASETS)
    p.add_argument('--models', nargs='+', choices=['NBC', 'GeFs'], default=['NBC', 'GeFs'])
    p.add_argument('--n-runs', type=int, default=5, help='random train/test splits per dataset')
    p.add_argument('--n-estimators', type=int, default=50, help='number of trees in the GeF')
    p.add_argument('--max-iter', type=int, default=100, help='max. bisection iterations (GeF robustness)')
    p.add_argument('--precision-bisearch', type=float, default=1e-9)
    p.add_argument('--n-splits', type=int, default=5, help='cross-validation folds (NBC)')
    p.add_argument('--n-bootstraps', type=int, default=10, help='bootstrap models for the epistemic/total uncertainty')
    p.add_argument('--fraction-test', type=float, default=0.4)
    p.add_argument('--max-size-test-set', type=int, default=500)
    p.add_argument('--results-dir', type=Path, default=REPO_ROOT / 'results' / 'pgm26')
    return p.parse_args(argv)


def run_model(model, X_train, y_train, X_test, y_test, C, F, args):
    if model == 'NBC':
        return standard_experiment_NBC(X_train, y_train, X_test, y_test, C, F, METRICS,
                                       n_splits=args.n_splits, N_bootstraps=args.n_bootstraps)
    return standard_experiment_GeFs(X_train, y_train, X_test, y_test, C, F, METRICS,
                                    n_estimators=args.n_estimators, N_bootstraps=args.n_bootstraps,
                                    precision_bisearch=args.precision_bisearch, max_iter=args.max_iter)


def main(argv=None):
    args = parse_args(argv)
    for sub in ['csv', 'accaccs', 'auc_summary']:
        (args.results_dir / sub).mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        np.random.seed(seed)
        print(f"\n--------------- Running experiment with seed: {seed} ---------------")

        for model in args.models:
            with open(args.results_dir / 'auc_summary' / f'{seed}_{model}.dat', 'w') as f:
                f.write("For each dataset, this table lists the reliability metrics sorted by their AUC "
                        "(shown in parentheses) from highest to lowest.\n")

        for set_name in args.datasets:
            print(f"\n--------------- {set_name} ---------------")

            X, y, X_test, y_test, C, F = load_dataset(set_name, randomize=True)
            N = int(X.shape[0] * args.fraction_test) if X_test is None else int(X_test.shape[0])
            print(f" with {C} classes, {F} features and {N} test points \n")

            robs = {m: [] for m in args.models}
            classifs = {m: [] for m in args.models}

            for _ in tqdm(range(args.n_runs)):
                X, y, X_test, y_test, C, F = load_dataset(set_name, randomize=True)
                X_train, y_train, X_test, y_test = split_train_test(
                    X, y, X_test, y_test, args.fraction_test, args.max_size_test_set)

                for model in args.models:
                    rob_metrics, classification_results, _ = run_model(
                        model, X_train, y_train, X_test, y_test, C, F, args)
                    robs[model].append(rob_metrics)
                    classifs[model].append(classification_results)

            for model in args.models:
                accuracies = np.mean(classifs[model], axis=1)
                print(f'Accuracy {model}: {np.mean(accuracies)} ± {np.std(accuracies)}')

                write_results_to_csv(np.vstack(classifs[model]), list(robs[model]), list_of_robm=True,
                                     set_name=f'{set_name}_{seed}_{model}', folder=str(args.results_dir / 'csv'))

                all_aucs = []
                with open(args.results_dir / 'accaccs' / f'{set_name}_{seed}_{model}.dat', 'w') as f:
                    for metric in METRICS:
                        accs, auc, _, stddev_auc = get_mean_accs(
                            robs[model], classifs[model], metric=metric, stddev=True)
                        all_aucs.append((auc, metric, stddev_auc))
                        f.write(f'fixed:{metric}\n')
                        f.write(','.join(str(x) for x in accs) + '\n')

                all_aucs.sort(key=lambda x: x[0], reverse=True)
                with open(args.results_dir / 'auc_summary' / f'{seed}_{model}.dat', 'a') as f:
                    f.write(f"\n{set_name:10.10s}:\t")
                    f.write(' | '.join(f'{metric:8.8s} ({auc:.5f}±{stddev:.5f})'
                                       for auc, metric, stddev in all_aucs) + '\n')


if __name__ == '__main__':
    main()
