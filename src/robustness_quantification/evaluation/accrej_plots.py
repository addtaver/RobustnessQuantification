import numpy as np
import matplotlib.pyplot as plt


def get_mean_accs(list_of_rob_metrics, list_of_classif_results, metric, stddev=False):
    list_of_accs = []
    list_of_aucs = []
    for i in range(len(list_of_rob_metrics)):
        accs = get_accaccs_curve(list_of_rob_metrics[i][metric], list_of_classif_results[i])
        list_of_accs.append(accs)
        list_of_aucs.append(np.mean(accs))

    list_of_accs = np.vstack(list_of_accs)
    list_of_aucs = np.vstack(list_of_aucs)

    accaccs = np.mean(list_of_accs, axis=0)
    auc = np.mean(list_of_aucs)

    if stddev:
        stddevs = np.std(list_of_accs, axis=0)
        stddevs_auc = np.std(list_of_aucs)
        return accaccs, auc, stddevs, stddevs_auc
    return accaccs, auc


def get_accaccs_curve(measure, classification_results, rejection=True, get_ordered_metrics=False):
	N = len(measure)
	measure_sorted = np.zeros(N)
	classification_results_sorted = np.zeros(N)

	# sort the data points by robustness measure for each of the models in the list
	arg_sorted_measure = np.argsort(measure)
	classification_results_sorted += classification_results[arg_sorted_measure]
	measure_sorted += measure[arg_sorted_measure]

	accs = np.zeros(N)

	# get the accuracy for each n
	for i in range(N):
		step_acc = classification_results_sorted[i:].sum()/(N-i)
		accs[i] = step_acc

	if not rejection:
		accs = accs[::-1]

	if get_ordered_metrics:
		return accs, measure_sorted, classification_results_sorted
	return accs


def hybrid_accaccs(metric_1, metric_2, classification_results, metric_3=None, mixing_factor=0.5):
	N = len(metric_1)

	ranks_metric_1 = np.empty_like(metric_1)
	ranks_metric_1[np.argsort(metric_1)] = np.arange(N)

	rank_metric_2 = np.empty_like(metric_2)
	rank_metric_2[np.argsort(metric_2)] = np.arange(N)

	hybrid_ranks = (mixing_factor * ranks_metric_1 * 1.0001 + (1-mixing_factor) * rank_metric_2)
	hybrid_sorted = np.argsort(hybrid_ranks)

	classification_results_sorted = np.zeros(N)
	classification_results_sorted += classification_results[hybrid_sorted]

	# initialize array to keep track of the accuracy for each n
	# the length of the array depends on the step size
	accs = np.zeros(N)

	# get the accuracy for each n
	for i in range(N):
		accs[i] = classification_results_sorted[i:].sum()/(N-i)

	# invert order of accs
	accs = accs[::-1]

	return accs

def hybrid_accaccs3(metric_1, metric_2, metric_3, classification_results, mixing_factor=0.5):
	N = len(metric_1)

	ranks_metric_1 = np.empty_like(metric_1)
	ranks_metric_1[np.argsort(metric_1)] = np.arange(N)

	rank_metric_2 = np.empty_like(metric_2)
	rank_metric_2[np.argsort(metric_2)] = np.arange(N)

	rank_metric_3 = np.empty_like(metric_3)
	rank_metric_3[np.argsort(metric_3)] = np.arange(N)

	hybrid_ranks = (mixing_factor[0] * ranks_metric_1 * 1.0001 + mixing_factor[1] * rank_metric_2 + mixing_factor[2] * rank_metric_3)
	hybrid_sorted = np.argsort(hybrid_ranks)

	classification_results_sorted = np.zeros(N)
	classification_results_sorted += classification_results[hybrid_sorted]

	# initialize array to keep track of the accuracy for each n
	# the length of the array depends on the step size
	accs = np.zeros(N)

	# get the accuracy for each n
	for i in range(N):
		accs[i] = classification_results_sorted[i:].sum()/(N-i)

	# invert order of accs
	accs = accs[::-1]

	return accs


def get_mean_accs_hybrid(list_of_rob_metrics, list_of_classif_results, metric, metric_2=None, metric_3=None, mixing=0.5, stddev=False, subset_indices=None):
    list_of_accs = []
    list_of_aucs = []

    for i in range(len(list_of_rob_metrics)):
        if metric_3 is not None:
            accs = hybrid_accaccs3(list_of_rob_metrics[i][metric], list_of_rob_metrics[i][metric_2], list_of_rob_metrics[i][metric_3], list_of_classif_results[i], mixing_factor=mixing)
        else:
            if subset_indices is not None:
                accs = hybrid_accaccs(list_of_rob_metrics[i][metric][subset_indices], list_of_rob_metrics[i][metric_2][subset_indices], list_of_classif_results[i][subset_indices], mixing_factor=mixing)
            else:
                accs = hybrid_accaccs(list_of_rob_metrics[i][metric], list_of_rob_metrics[i][metric_2], list_of_classif_results[i], mixing_factor=mixing)

        list_of_accs.append(accs)
        list_of_aucs.append(np.mean(accs))

    list_of_accs = np.vstack(list_of_accs)
    list_of_aucs = np.vstack(list_of_aucs)

    accaccs = np.mean(list_of_accs, axis=0)
    auc = np.mean(list_of_aucs)

    if stddev:
        stddevs = np.std(list_of_accs, axis=0)
        stddevs_auc = np.std(list_of_aucs)
        return accaccs, auc, stddevs, stddevs_auc
    return accaccs, auc


def get_ideal_accaccs(classification_results):
	N = len(classification_results)

	# compute ideal curve: all correct predictions first
	ideal_curve = np.zeros(N)
	ideal_curve[:int(classification_results.sum())] = 1
	ideal_accs = np.cumsum(ideal_curve)/np.arange(1, N+1)

	return ideal_accs


def accuracy_rejection_curve_single(measure, classification_results, ax=None, set_name=None, key=None, color=None):
    """
    Plot the mean accuracy-rejection curve for all models.
    The x-axis is the rejection rate and the y-axis is the accuracy.
    If split is True, the data points in each window are split by class.
    measure is an array of shape (MxN) that contains for each of the M models, the robustness measure for all N samples
    y contains the true labels for the N samples
    classification_results is an array of shape (MxN) that contains for each of the models the clasification result for each of the samples.
    The classification result is 1 if the prediction was right, and 0 otherwise
    """
    N = len(measure)
    accs = get_accaccs_curve(measure, classification_results)

    ### plot the figure ###
    # if no axis is given, create a new figure
    # else plot on the given axis
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 4))
        if key is not None:
            ax.set_title(key)

    if color is None:
        color = 'green'


    ax.plot(np.linspace(0, 100, N), accs, color=color)
    ax.set_xlabel("Percentage of accepted samples")
    ax.set_ylabel("Accuracy over the accepted samples")

    if not ax:
        # save plot
        if set_name:
            plt.savefig(f"plots/{set_name}/{set_name}_acc_rej_rob.png")
        plt.show()



def accuracy_rejection_curve_single_AUC(measure, classification_results, ax=None, set_name=None, key=None, color=None):
    """
    Plot the mean accuracy-rejection curve for all models.
    The x-axis is the rejection rate and the y-axis is the accuracy.
    If split is True, the data points in each window are split by class.
    measure is an array of shape (MxN) that contains for each of the M models, the robustness measure for all N samples
    y contains the true labels for the N samples
    classification_results is an array of shape (MxN) that contains for each of the models the clasification result for each of the samples.
    The classification result is 1 if the prediction was right, and 0 otherwise
    """
    N = len(measure)
    accs = get_accaccs_curve(measure, classification_results)

    auc = np.mean(accs)
    print("\nAUC: ", auc)

    ### plot the figure ###
    # if no axis is given, create a new figure
    # else plot on the given axis
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 4))
        if key is not None:
            ax.set_title(key)

    if color is None:
        color = 'green'


    ax.plot(np.linspace(0, 100, N), accs, color=color)
    ax.set_xlabel("Percentage of accepted samples")
    ax.set_ylabel("Accuracy over the accepted samples")

    if not ax:
        # save plot
        if set_name:
            plt.savefig(f"plots/{set_name}/{set_name}_acc_rej_rob.png")
        plt.show()

    return auc


def accuracy_rejection_curve(measure, classification_results, percentiles=None, ax=None, set_name=None, mean=False, stddev=False, key=None, color=None, AUC=False):
    """
    Plot the mean accuracy-rejection curve for all models.
    The x-axis is the rejection rate and the y-axis is the accuracy.
    If split is True, the data points in each window are split by class.
    measure is an array of shape (MxN) that contains for each of the M models, the robustness measure for all N samples
    y contains the true labels for the N samples
    classification_results is an array of shape (MxN) that contains for each of the models the clasification result for each of the samples.
    The classification result is 1 if the prediction was right, and 0 otherwise
    """
    if len(measure.shape) == 1:
        if AUC:
            return accuracy_rejection_curve_single_AUC(measure, classification_results, ax=ax, set_name=set_name, key=key, color=color)
        accuracy_rejection_curve_single(measure, classification_results, ax=ax, set_name=set_name, key=key, color=color)
        return

    M = len(measure)
    N = len(measure[0])

    measure_bis = np.zeros(measure.shape)
    classification_results_bis = np.zeros(classification_results.shape)

    # sort the data points by robustness measure for each of the models in the list
    for i in range(M):
        S = sorted(zip(measure[i], classification_results[i]), key=lambda x: x[0])
        measure_bis[i] = np.array([x[0] for x in S])
        classification_results_bis[i] = np.array([x[1] for x in S])


    # initialize array to keep track of the accuracy for each n
    # the length of the array depends on the step size
    accs = np.zeros(N)
    if percentiles is not None:
        percents = np.zeros((N, 2))
    if stddev:
        stddevs = np.zeros(N)

    # get the accuracy for each n
    for i in range(N):
        step_accs = []
        for j in range(M):
            step_acc = classification_results_bis[j][i:].sum()/(N-i)
            step_accs.append(step_acc)
        accs[i] = np.mean(step_accs)
        if stddev:
            stddevs[i] = np.std(step_accs)
        if percentiles is not None:
            percents[i] = np.percentile(step_accs, percentiles)

    ### plot the figure ###
    # if no axis is given, create a new figure
    # else plot on the given axis

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(7, 4))
        if key is not None:
            ax.set_title(key)

    if color is None:
        color = 'green'

    if mean:
        ax.plot(np.linspace(0, 100, N), accs[::-1], color=color)
        if stddev:
            ax.fill_between(np.linspace(0, 100, N), accs[::-1]-2*stddevs[::-1], accs[::-1]+2*stddevs[::-1], alpha=0.5, color=color)
    elif stddev:
        ax.plot(np.linspace(0, 100, N), stddevs[::-1], color=color)
    if percentiles is not None:
        ax.plot(np.linspace(0, 100, N), percents[::-1][:, 0], color=color)
        ax.plot(np.linspace(0, 100, N), percents[::-1][:, 1], color=color)
        ax.fill_between(np.linspace(0, 100, N), percents[::-1][:, 0], percents[::-1][:, 1], alpha=0.3, color=color)
    # ax.plot(np.arange(0, 100, 100/N), accs)
    # set axis labels
    ax.set_xlabel("Percentage of accepted samples")
    ax.set_ylabel("Accuracy over the accepted samples")


    if not ax:
        # save plot
        if set_name:
            plt.savefig(f"plots/{set_name}/{set_name}_acc_rej_rob.png")
        plt.show()



Colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

def combine_plots_acc_rej(rob_measures, classification_results, set_name=None, fig_size=None, percentiles=None, mean=False, stddev=False, keys=None, show=True, title_extra="", folder=None, y_min=0.75, y_max=0.35, legend_loc='upper right', colors=Colors, AUC=False):
    if fig_size is None:
        fig_size = (8,2)

    fig, axs = plt.subplots(1, 1, figsize=fig_size)
    axs = [axs]

    counter = 0
    for rob_key in rob_measures:
        if keys is not None:
            if rob_key not in keys:
                continue

        accuracy_rejection_curve(rob_measures[rob_key], classification_results,  ax=axs[0], set_name=None, percentiles=percentiles, color=colors[counter], mean=mean, stddev=stddev, AUC=AUC)
        counter += 1

    extra = ""
    if mean:
        extra += " mean"
    if stddev:
        extra += " std dev"
    if percentiles is not None:
        extra += " percentiles"
    # title of the plot
    fig.suptitle("Accuracy-rejection curves" + extra + " " + title_extra)

    # titles and legends of the subplots
    # axs[0].set_title("All points")
    if keys is not None:
        l = keys
    else:
        l = list(rob_measures.keys())

    legend_location = legend_loc
    if not mean and stddev:
        if legend_loc != 'empty':
            legend_location = 'upper right'
        plt.ylim(0, y_max)
    else:
        plt.ylim(y_min, 1)

    if legend_loc != 'empty':
        if colors is not None:
            axs[0].legend(l, labelcolor=colors, loc=legend_location)
            leg = axs[0].get_legend()
            for i in range(len(l)):
                leg.legendHandles[i].set_color(colors[i])
        else:
            axs[0].legend(l, loc=legend_location)

    # plt.tight_layout()
    plt.xlim(0, 100)

    if set_name:
        if folder is None:
            folder = 'plots'
        # robs = '_'.join(list(rob_measures.keys()))
        # plt.savefig(f"plots/{set_name}/{set_name}_{robs}_acc_rej.png")
        plt.savefig(f"{folder}/{set_name}.png")

    if show:
        plt.show()
    else:
        plt.close()