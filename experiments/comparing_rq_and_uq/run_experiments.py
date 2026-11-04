"""Run the comparing_rq_and_uq experiments: reliability metrics (uncertainty, global/local epsilon-robustness and hybrids of both)
for NBCs and GeFs, trained on small (optionally label-noised) training sets.

Writes, per dataset/seed/training size/noise level (``name``), in ``<results-dir>``:
    csv/<name>_<model>.csv                raw classification results and metrics
    accaccs/<name>_<model>.dat            accuracy-rejection curves (fixed/uncertainty/hybrid metrics)
    auc/<name>_<model>.dat                AUCs of the metrics and hybrids + rankings
    summaries/<dataset>_<seed>_<model>_combined.dat   AUC (stddev) of every metric across noise levels (columns)
                                                      and training sizes (rows), written once all sizes are done
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from robustness_quantification.data.loading import load_dataset
from robustness_quantification.data.splits import introduce_label_noise, split_train_test
from robustness_quantification.evaluation.accrej_plots import get_mean_accs
from robustness_quantification.evaluation.hybrid import find_best_hybrid_metric, find_optimal_metric, normalize_metrics
from robustness_quantification.evaluation.store_results import write_results_to_csv
from robustness_quantification.pipelines.gef import standard_experiment_on_train_GeFs
from robustness_quantification.pipelines.nbc import standard_experiment_on_train_NBC

REPO_ROOT = Path(__file__).resolve().parents[2]

DATASETS = ['Australian', 'adult', 'bank', 'BCW', 'german', 'Healthy_aging', 'heart_disease', 'lymphography',
            'nursery', 'solar_big', 'solar_small', 'spect_heart', 'student_math_bin', 'student_port_bin']
UNCERTAINTY_METRICS = ['max_prob', 'entropies', 'aleatoric', 'epistemic', 'total', 'margin_of_confidence']
ROBUSTNESS_METRICS = ['global_eps', 'local_eps']
METRICS = UNCERTAINTY_METRICS + ROBUSTNESS_METRICS
# order in which uncertainty metrics are combined with the robustness metrics (determines the output file layout)
HYBRID_ORDER = ['margin_of_confidence', 'max_prob', 'aleatoric', 'total', 'epistemic', 'entropies']
MODELS = ['NBC', 'GeFs']
NOISE_LABELS = []  # noise factors of the current run, set in main()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--datasets', nargs='+', default=DATASETS)
    p.add_argument('--training-sizes', nargs='+', default=['50', '100', '200'],
                   help="training set sizes, 'ALL' uses the complete training set")
    p.add_argument('--noise-factors', type=float, nargs='+', default=[0.0, 0.10, 0.20])
    p.add_argument('--n-runs', type=int, default=7, help='random train/test splits per training size')
    p.add_argument('--n-estimators', type=int, default=30, help='number of trees in the GeF')
    p.add_argument('--max-iter', type=int, default=100)
    p.add_argument('--precision-bisearch', type=float, default=1e-9)
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--n-bootstraps', type=int, default=10)
    p.add_argument('--fraction-test', type=float, default=0.4)
    p.add_argument('--max-size-test-set', type=int, default=1000)
    p.add_argument('--results-dir', type=Path, default=REPO_ROOT / 'results' / 'comparing_rq_and_uq')
    return p.parse_args(argv)


def run_model(model, X_t, y_t, X_test, y_test, C, F, args):
    if model == 'NBC':
        return standard_experiment_on_train_NBC(X_t, y_t, X_test, y_test, C, F, METRICS,
                                                n_splits=args.n_splits, N_bootstraps=args.n_bootstraps)
    return standard_experiment_on_train_GeFs(X_t, y_t, X_test, y_test, C, F, METRICS,
                                             n_estimators=args.n_estimators, N_bootstraps=args.n_bootstraps,
                                             precision_bisearch=args.precision_bisearch, max_iter=args.max_iter)


def collect_runs(set_name, size, noise_factor, args):
    """Run all repetitions for one configuration. Returns per model: test/train metrics and classification results."""
    runs = {m: dict(rob=[], classif=[], rob_train=[], classif_train=[]) for m in MODELS}

    for _ in tqdm(range(args.n_runs)):
        X, y, X_test, y_test, C, F = load_dataset(set_name, randomize=True)
        X_train, y_train, X_test, y_test = split_train_test(
            X, y, X_test, y_test, args.fraction_test, args.max_size_test_set)

        if noise_factor > 0:
            y_train = introduce_label_noise(y_train, C, noise_fraction=noise_factor)

        if size is not None:
            perm = np.random.permutation(len(y_train))
            X_t, y_t = X_train[perm][:size], y_train[perm][:size]
        else:
            X_t, y_t = X_train, y_train

        for model in MODELS:
            rob, classif, _, rob_train, classif_train, _ = run_model(model, X_t, y_t, X_test, y_test, C, F, args)
            runs[model]['rob'].append(rob)
            runs[model]['classif'].append(classif)
            runs[model]['rob_train'].append(rob_train)
            runs[model]['classif_train'].append(classif_train)

    return runs


def evaluate_model(runs, rob_shifted):
    """Compute curves/AUCs of all metrics and hybrids for one model. Returns the text of the accaccs and auc files."""
    rob, classif = runs['rob'], runs['classif']
    rob_norm = normalize_metrics(rob, METRICS)
    rob_train_norm = normalize_metrics(runs['rob_train'], METRICS)

    accaccs_txt, auc_txt = [], []
    all_aucs, opt_aucs = [], []
    first_fixed = ROBUSTNESS_METRICS[0]

    for metric_fixed in ROBUSTNESS_METRICS:
        accs_fixed, auc_fixed, _, stddev_auc_fixed = get_mean_accs(rob, classif, metric=metric_fixed, stddev=True)
        all_aucs.append((auc_fixed, metric_fixed))
        opt_aucs.append((auc_fixed, metric_fixed))
        rob_shifted[metric_fixed].append((auc_fixed, stddev_auc_fixed))
        accaccs_txt.append(f'fixed:{metric_fixed}\n' + ','.join(str(x) for x in accs_fixed) + '\n')

        for metric in HYBRID_ORDER:
            accs_metric, auc_metric, _, stddev_auc_metric = get_mean_accs(rob, classif, metric=metric, stddev=True)
            if metric_fixed == first_fixed:
                all_aucs.append((auc_metric, metric))
                opt_aucs.append((auc_metric, metric))
                rob_shifted[metric].append((auc_metric, stddev_auc_metric))

            best_mixing, hybrid_accs, auc_hybrid, _, _ = find_best_hybrid_metric(
                rob_norm, rob_train_norm, classif, runs['classif_train'],
                metric_1=metric, metric_2=metric_fixed, stddev=True, hybrid_to_compute='new')
            all_aucs.append((auc_hybrid, f'{metric_fixed}_{metric}_{best_mixing:.2f}'))

            optimal_mixing, optimal_auc = find_optimal_metric(rob_norm, classif, metric_1=metric, metric_2=metric_fixed)
            opt_aucs.append((optimal_auc, f'{metric_fixed}_{metric}_{optimal_mixing:.2f}'))

            if metric_fixed == first_fixed:
                accaccs_txt.append(f'uncer:{metric}\n' + ','.join(str(x) for x in accs_metric) + '\n')
            accaccs_txt.append(f'hybri:{metric_fixed},{metric}\n' + ','.join(str(x) for x in hybrid_accs) + '\n')

            auc_txt.append(f'AUC {metric_fixed:7.7} vs {metric:5.5} vs Hybrid {best_mixing:.2f} ({optimal_mixing:.2f}): '
                           f'{auc_fixed:.8f} {auc_metric:.8f} {auc_hybrid:.8f} ({optimal_auc:.8f})\n\n')
        auc_txt.append('\n')

    all_aucs.sort(key=lambda x: x[0], reverse=True)
    opt_aucs.sort(key=lambda x: x[0], reverse=True)
    empty = ' '
    auc_txt.append(f'\nRanking of all metrics and combinations by AUC: {empty:44s} '
                   f'Ranking of all metrics and OPTIMAL combinations by AUC:\n\n')
    for (auc, name), (opt_auc, opt_name) in zip(all_aucs, opt_aucs):
        auc_txt.append(f'{name:60s} {auc:.8f} {empty:20s} {opt_name:60s} {opt_auc:.8f}\n')

    return ''.join(accaccs_txt), ''.join(auc_txt)


def write_summary(path, per_size, col_width=25):
    """Write AUC (stddev) of every metric: one block per metric, one row per training size, one column per noise level."""
    sizes = sorted(per_size, key=lambda s: float('inf') if s.upper() == 'ALL' else int(s), reverse=True)
    labels = {s: f'Ndata={s}' for s in sizes}
    w = max(len(m) for m in METRICS + list(labels.values()))
    lines = []
    for metric in METRICS:
        n_levels = len(per_size[sizes[0]][metric])
        header = ''.join(f'{f"{round(100 * nf)}% DS":<{col_width}}' for nf in NOISE_LABELS[:n_levels])
        lines.append(f'{metric:<{w}} : {header}'.rstrip())
        for s in sizes:
            cells = ''.join(f'{f"{auc:.8f} ({std:.8f})":<{col_width}}' for auc, std in per_size[s][metric])
            lines.append(f'{labels[s]:<{w}} : {cells}'.rstrip())
        lines.append('')
    path.write_text('\n'.join(lines))


def main(argv=None):
    args = parse_args(argv)
    for sub in ['csv', 'accaccs', 'auc', 'summaries']:
        (args.results_dir / sub).mkdir(parents=True, exist_ok=True)
    NOISE_LABELS[:] = args.noise_factors
    summary = defaultdict(dict)

    np.random.seed(args.seed)
    print(f"\n--------------- Running experiment with seed: {args.seed} ---------------")

    for size_arg in args.training_sizes:
        size = None if size_arg.upper() == 'ALL' else int(size_arg)
        size_suffix = '_ALL' if size is None else f'_{size}'

        for set_name in args.datasets:
            print(f"\n ------- {set_name} (training size {size_arg}) -------\n")
            base_name = f'{set_name}_{args.seed}{size_suffix}'
            rob_shifted = {m: {k: [] for k in METRICS} for m in MODELS}

            for noise_factor in args.noise_factors:
                name = f'{base_name}_noise_{noise_factor}'
                runs = collect_runs(set_name, size, noise_factor, args)

                for model in MODELS:
                    print(f'Accuracy {model}: ', np.mean(runs[model]['classif']))
                    write_results_to_csv(np.vstack(runs[model]['classif']), list(runs[model]['rob']),
                                         list_of_robm=True, set_name=f'{name}_{model}',
                                         folder=str(args.results_dir / 'csv'))

                    accaccs_txt, auc_txt = evaluate_model(runs[model], rob_shifted[model])
                    (args.results_dir / 'accaccs' / f'{name}_{model}.dat').write_text(accaccs_txt)
                    (args.results_dir / 'auc' / f'{name}_{model}.dat').write_text(auc_txt)

            for model in MODELS:
                summary[(set_name, model)][size_arg] = rob_shifted[model]

    for (set_name, model), per_size in summary.items():
        write_summary(args.results_dir / 'summaries' / f'{set_name}_{args.seed}_{model}_combined.dat', per_size)


if __name__ == '__main__':
    main()
