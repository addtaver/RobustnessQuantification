import numpy as np


def write_results_to_csv(classification_results, rob_metrics, set_name, list_of_robm=False, delimiter=',', folder='', file_number=None):
    # write function to write results to csv file
    if delimiter == ',':
        suffix = 'csv'
    else:
        suffix = 'dat'

    if file_number is not None:
        set_name = f"{set_name}_{file_number}"
    filename = f'{folder}/{set_name}.{suffix}'

    if len(classification_results.shape) == 1:
        n_runs = 1
    else:
        n_runs = classification_results.shape[0]

    with open(filename, 'w') as f:
        # write first line with column names
        f.write(f'n_runs={n_runs}\n\n')

        f.write(f'Classification Results\n')


        if len(classification_results.shape) == 1:
            classification_results = classification_results.reshape(1, -1)

        for classification_row in classification_results:
            f.write(delimiter.join([str(int(x)) for x in classification_row]) + '\n')


        if not list_of_robm:
            rob_metrics_list = [rob_metrics]
        else:
            rob_metrics_list = rob_metrics

        for r in rob_metrics_list[0].keys():
            f.write(f'Metric: {r}\n')

            for run in range(n_runs):

                if len(rob_metrics_list[run][r].shape) == 1:
                    rob_m = rob_metrics_list[run][r].copy()
                    rob_m = rob_m.reshape(1, -1)

                for rob_row in rob_m:
                    f.write(delimiter.join([str(x) for x in rob_row]) + '\n')


def read_results_from_csv(filename, delimiter=','):
    # read function to read results from csv file
    with open(filename, 'r') as f:
        lines = f.readlines()
        classification_results = []
        rob_metrics = {}
        current_metric = ''
        for line in lines:
            if line == 'Classification Results\n':
                current_metric = 'classification'
                continue
            if line.strip()[:6] == 'Metric':
                if current_metric == 'classification':
                    classification_results = np.array(classification_results)
                else:
                    rob_metrics[current_metric] = np.array(rob_metrics[current_metric])
                current_metric = line.strip()[8:]
                rob_metrics[current_metric] = []
                continue
            if current_metric == 'classification':
                classification_results.append(np.array([int(x) for x in line.strip().split(delimiter)]))
            else:
                rob_metrics[current_metric].append(np.array([float(x) for x in line.strip().split(delimiter)]))
        rob_metrics[current_metric] = np.array(rob_metrics[current_metric])

    return classification_results, rob_metrics

def read_accaccs(filename):
    """Read an accuracy-rejection file written by the experiment scripts.

    The file alternates a header line ('fixed:<metric>', 'uncer:<metric>' or 'hybri:<m1>,<m2>') with the
    comma-separated curve. Returns a dict mapping the metric name (text after the colon) to the curve as a
    list of strings.
    """
    curves = {}
    metric = None
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if metric is not None:
                curves[metric] = line.split(',')
                metric = None
            elif line[:5] in ('fixed', 'uncer', 'hybri') and ':' in line:
                metric = line.split(':')[1].strip()
    return curves
