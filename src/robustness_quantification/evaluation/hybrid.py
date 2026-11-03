"""Hybrid (convex combination) reliability metrics, as used in the UAI experiments."""
import numpy as np

from robustness_quantification.evaluation.accrej_plots import get_accaccs_curve, get_mean_accs_hybrid

REVERSE_ORDER_METRICS = ['entropies', 'aleatoric', 'epistemic', 'total']


def get_mean_accs_new_hybrid(list_of_rob_metrics, list_of_classif_results, metric_1, metric_2, mixing, stddev=False):
    list_of_accs = []
    list_of_aucs = []

    for rob_metrics, classif_results in zip(list_of_rob_metrics, list_of_classif_results):
        hybrid = mixing * rob_metrics[metric_1] + (1 - mixing) * rob_metrics[metric_2]

        accs = get_accaccs_curve(hybrid, classif_results)
        list_of_accs.append(accs)
        list_of_aucs.append(np.mean(accs))

    list_of_accs = np.vstack(list_of_accs)
    list_of_aucs = np.vstack(list_of_aucs)

    accaccs = np.mean(list_of_accs, axis=0)
    auc = np.mean(list_of_aucs)

    if stddev:
        return accaccs, auc, np.std(list_of_accs, axis=0), np.std(list_of_aucs)
    return accaccs, auc


def _hybrid_auc(list_of_rob_metrics, list_of_classif_results, metric_1, metric_2, mixing, hybrid_to_compute):
    if hybrid_to_compute == 'new':
        _, auc = get_mean_accs_new_hybrid(list_of_rob_metrics, list_of_classif_results, metric_1=metric_1, metric_2=metric_2, mixing=mixing, stddev=False)
    else:
        _, auc = get_mean_accs_hybrid(list_of_rob_metrics, list_of_classif_results, metric_1, metric_2=metric_2, mixing=mixing)
    return auc


def find_best_hybrid_metric(list_of_rob_metrics, list_of_rob_metrics_train, list_of_classif_results, list_of_classif_results_train, metric_1, metric_2, stddev=True, hybrid_to_compute=None):
    """Select the mixing weight on the training data, and evaluate the hybrid on the test data."""
    best_mixing = 0.0
    best_auc = -100
    for mixing in np.linspace(0.0, 1.0, num=101):
        auc_hybrid_train = _hybrid_auc(list_of_rob_metrics_train, list_of_classif_results_train, metric_1, metric_2, mixing, hybrid_to_compute)
        if auc_hybrid_train > best_auc:
            best_auc = auc_hybrid_train
            best_mixing = mixing

    if hybrid_to_compute == 'new':
        result = get_mean_accs_new_hybrid(list_of_rob_metrics, list_of_classif_results, metric_1=metric_1, metric_2=metric_2, mixing=best_mixing, stddev=stddev)
    else:
        result = get_mean_accs_hybrid(list_of_rob_metrics, list_of_classif_results, metric_1, metric_2=metric_2, mixing=best_mixing, stddev=stddev)

    return (best_mixing, *result)


def find_optimal_metric(list_of_rob_metrics, list_of_classif_results, metric_1, metric_2, hybrid_to_compute=None):
    """Select the mixing weight that is optimal on the given (test) data."""
    opt_mixing = 0.0
    opt_auc = -100
    for mixing in np.linspace(0.0, 1.0, num=101):
        auc_hybrid = _hybrid_auc(list_of_rob_metrics, list_of_classif_results, metric_1, metric_2, mixing, hybrid_to_compute)
        if auc_hybrid > opt_auc:
            opt_auc = auc_hybrid
            opt_mixing = mixing

    return opt_mixing, opt_auc


def normalize_metrics(list_of_rob_metrics, metrics_to_compute):
    """Min-max normalise every metric to [0, 1] such that higher always means more reliable."""
    list_of_rob_metrics_norm = [rob_metric.copy() for rob_metric in list_of_rob_metrics]

    for metric in metrics_to_compute:
        for rob_metrics in list_of_rob_metrics_norm:
            values = rob_metrics[metric].copy()
            if metric in REVERSE_ORDER_METRICS:
                values *= -1
            R = (values - np.min(values)) / (np.max(values) - np.min(values))
            if metric in REVERSE_ORDER_METRICS:
                R = 1 - R
            rob_metrics[metric] = R

    return list_of_rob_metrics_norm
