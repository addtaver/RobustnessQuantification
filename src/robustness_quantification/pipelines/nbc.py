from sklearn.model_selection import KFold, StratifiedKFold

import numpy as np

from NBC import NBC, calculate_max_prob, calculate_entropy, calculate_precise_diff, calculate_epistimic_uncertainty_bootstrapped


def cross_validation(X, y, C, feat_vals, n_splits=3, stratified=True):
	# skf = KFold(n_splits=n_splits, shuffle=True)
	if stratified:
		skf = StratifiedKFold(n_splits=n_splits, shuffle=True)
	else:
		skf = KFold(n_splits=n_splits, shuffle=True)

	max_accuracy = 0
	best_alpha = 0


	alphas = [0.0001, 0.001, 0.01, 0.1, 1]

	for alph in alphas:
		accuracy = 0
		for _, (train_index, test_index) in enumerate(skf.split(X, y)):
			nb = NBC(alpha=alph)

			X_train, X_test = X[train_index], X[test_index]
			y_train, y_test = y[train_index], y[test_index]

			# fit the model to the training data
			nb.fit(X_train, y_train, feat_vals=feat_vals, classes=np.arange(C))

			# predict the class of the test
			pred_prob = nb.predict_posterior(X_test)
			y_pred = nb.classes[np.argmax(pred_prob, axis=1)]

			accuracy += np.sum(y_test == y_pred) / len(y_test)

		accuracy /= n_splits
		if accuracy > max_accuracy:
			max_accuracy = accuracy
			best_alpha = alph

	return best_alpha


def standard_experiment_NBC(X_train, y_train, X_test, y_test, C, F, metrics_to_calc, n_splits=5, stratified=False, N_bootstraps=10, max_iter=40):
	"""
		This experiment fits an NBC to a dataset, predicts the classes of a test set and the corresponding probabilities, and then computes all the reliability metrics on the test set.

		Parameters
		----------
		X_train: numpy array n x m
			Training data with n samples and m variables.
		y_train: numpy array n x 1
			Training labels with n samples and 1 variable.
		X_test: numpy array n x m
			Test data with n samples and m variables.
		y_test: numpy array n x 1
			Test labels with n samples and 1 variable.
		C: int
			Number of classes.
		F: list of int
			Number of feature values for each feature.
		metrics_to_calc: list of str
			List of metrics to calculate. Possible values are 'max_prob', 'entropies', 'aleatoric', 'epistemic', 'total', 'margin_of_confidence', 'global_eps', 'local_eps', 'total_var', 'chi2'.
		n_splits: int
			Number of splits for cross-validation to determine the optimal smoothing parameter alpha.
		stratified: bool
			Whether to use stratified cross-validation or not.
		N_bootstraps: int
			Number of bootstraps to use for calculating the ensemble uncertainty metrics.
		max_iter: int
			Maximum number of iterations for the bisection search when computing the robustness metrics.

		Returns
		-------
		rob_metrics: dict
			Dictionary containing the calculated metrics for each instance in the test set.
		classification_results: numpy array of shape (n,)
			Binary array indicating whether the predicted class matches the true class for each instance in the test set.
		y_pred: numpy array of shape (n,)
			The predicted class for each instance in the test set.
	"""

	N_test = X_test.shape[0]

	# initialize the arrays to store the results
	rob_metrics = {}
	for _, metric in enumerate(metrics_to_calc):
		rob_metrics[metric] = np.zeros(N_test)
		if metric == 'margin_of_confidence':
			rob_metrics['global_eps'] = np.zeros(N_test)


	classification_results = np.zeros(N_test)
	y_pred = np.zeros(N_test)


	# this array contains the possible values for each feature
	feat_vals = [np.arange(F[i]) for i in range(len(F))]

	# determine the optimal smoothing parameter alpha using cross-validation
	alpha = cross_validation(X_train, y_train, C, feat_vals, n_splits=n_splits, stratified=stratified)

	# fit the model to the training data
	nb = NBC(alpha=alpha)
	nb.fit(X_train, y_train, feat_vals=feat_vals, classes=np.arange(C))

	pred_prob = nb.predict_posterior(X_test)
	y_pred += nb.classes[np.argmax(pred_prob, axis=1)]
	classification_results += (y_test == y_pred).astype(float)
	### calculate the reliability metrics

	## uncertainty metrics:
	if 'max_prob' in metrics_to_calc:
		# probability of the predicted class (highest one) for each data point
		rob_metrics['max_prob'] += calculate_max_prob(pred_prob)

	if 'entropies' in metrics_to_calc:
		# entropy of the predicted probabilities for each data point
		rob_metrics['entropies'] -= calculate_entropy(pred_prob)

	# ensemble uncertainty metrics
	if 'aleatoric' in metrics_to_calc or 'epistemic' in metrics_to_calc or 'total' in metrics_to_calc:
		avg_entropy, entropy_avg, _ = calculate_epistimic_uncertainty_bootstrapped(N_bootstraps, X_train, y_train, X_test, alpha, feat_vals=feat_vals)
		if 'aleatoric' in metrics_to_calc:
			rob_metrics['aleatoric'] += avg_entropy
		if 'total' in metrics_to_calc:
			rob_metrics['total'] += entropy_avg
		if 'epistemic' in metrics_to_calc:
			rob_metrics['epistemic'] += entropy_avg - avg_entropy

	if 'margin_of_confidence' in metrics_to_calc or 'global_eps' in metrics_to_calc:
		# difference between the highest and second highest probability for each data point
		_, diff, global_eps = calculate_precise_diff(pred_prob, global_eps=True)
		if 'margin_of_confidence' in metrics_to_calc:
			rob_metrics['margin_of_confidence'] += diff

		# global epsilon (robustness metric), is a scaled version of the deltas
		if 'global_eps' in metrics_to_calc:
			rob_metrics['global_eps'] += global_eps


	## robustness metrics:
	if 'local_eps' in metrics_to_calc:
		rob_metrics['local_eps'] += nb.predict_local_epsilon_rob(X_test, step_size=1e-9, upper_bound=1.0, max_iter=max_iter)

	if 'total_var' in metrics_to_calc:
		rob_metrics['total_var'] += nb.predict_local_TV_rob(X_test, max_iter=max_iter, min_rob=1e-10)

	if 'chi2' in metrics_to_calc:
		rob_metrics['chi2'] += nb.predict_local_chi2_rob(X_test, step_size=1e-9, upper_bound=1.0, max_iter=max_iter)


	return rob_metrics, classification_results, y_pred


def standard_experiment_on_train_NBC(X_train, y_train, X_test, y_test, C, F, metrics_to_calc, n_splits=5, stratified=False, N_bootstraps=10, max_iter=40):
	"""
		This experiment trains the model on a dataset, and then computes all the reliability metrics on a test set.
	"""

	N_test = X_test.shape[0]
	N_train = X_train.shape[0]

	# initialize the arrays to store the results
	rob_metrics = {}
	rob_metrics_train = {}
	for _, metric in enumerate(metrics_to_calc):
		rob_metrics[metric] = np.zeros(N_test)
		rob_metrics_train[metric] = np.zeros(N_train)
		if metric == 'margin_of_confidence':
			rob_metrics['global_eps'] = np.zeros(N_test)
			rob_metrics_train['global_eps'] = np.zeros(N_train)


	classification_results = np.zeros(N_test)
	y_pred = np.zeros(N_test)

	classification_results_train = np.zeros(N_train)
	y_pred_train = np.zeros(N_train)


	# this array contains the possible values for each feature
	feat_vals = [np.arange(F[i]) for i in range(len(F))]

	# determine the optimal smoothing parameter alpha using cross-validation
	alpha = cross_validation(X_train, y_train, C, feat_vals, n_splits=n_splits, stratified=stratified)

	# fit the model to the training data
	nb = NBC(alpha=alpha)
	nb.fit(X_train, y_train, feat_vals=feat_vals, classes=np.arange(C))

	pred_prob = nb.predict_posterior(X_test)
	y_pred += nb.classes[np.argmax(pred_prob, axis=1)]
	classification_results += (y_test == y_pred).astype(float)

	pred_prob_train = nb.predict_posterior(X_train)
	y_pred_train += nb.classes[np.argmax(pred_prob_train, axis=1)]
	classification_results_train += (y_train == y_pred_train).astype(float)
	### calculate the reliability metrics

	## uncertainty metrics:
	if 'max_prob' in metrics_to_calc:
		# probability of the predicted class (highest one) for each data point
		rob_metrics['max_prob'] += calculate_max_prob(pred_prob)
		rob_metrics_train['max_prob'] += calculate_max_prob(pred_prob_train)

	if 'entropies' in metrics_to_calc:
		# entropy of the predicted probabilities for each data point
		rob_metrics['entropies'] -= calculate_entropy(pred_prob)
		rob_metrics_train['entropies'] -= calculate_entropy(pred_prob_train)

	# ensemble uncertainty metrics
	if 'aleatoric' in metrics_to_calc or 'epistemic' in metrics_to_calc or 'total' in metrics_to_calc:
		avg_entropy, entropy_avg, _ = calculate_epistimic_uncertainty_bootstrapped(N_bootstraps, X_train, y_train, X_test, alpha, feat_vals=feat_vals)
		avg_entropy_train, entropy_avg_train, _ = calculate_epistimic_uncertainty_bootstrapped(N_bootstraps, X_train, y_train, X_train, alpha, feat_vals=feat_vals)
		if 'aleatoric' in metrics_to_calc:
			rob_metrics['aleatoric'] += avg_entropy
			rob_metrics_train['aleatoric'] += avg_entropy_train
		if 'total' in metrics_to_calc:
			rob_metrics['total'] += entropy_avg
			rob_metrics_train['total'] += entropy_avg_train
		if 'epistemic' in metrics_to_calc:
			rob_metrics['epistemic'] += entropy_avg - avg_entropy
			rob_metrics_train['epistemic'] += entropy_avg_train - avg_entropy_train

	if 'margin_of_confidence' in metrics_to_calc or 'global_eps' in metrics_to_calc:
		# difference between the highest and second highest probability for each data point
		_, diff, global_eps = calculate_precise_diff(pred_prob, global_eps=True)
		_, diff_train, global_eps_train = calculate_precise_diff(pred_prob_train, global_eps=True)
		if 'margin_of_confidence' in metrics_to_calc:
			rob_metrics['margin_of_confidence'] += diff
			rob_metrics_train['margin_of_confidence'] += diff_train

		# global epsilon (robustness metric), is a scaled version of the deltas
		if 'global_eps' in metrics_to_calc:
			rob_metrics['global_eps'] += global_eps
			rob_metrics_train['global_eps'] += global_eps_train


	## robustness metrics:
	if 'local_eps' in metrics_to_calc:
		rob_metrics['local_eps'] += nb.predict_local_epsilon_rob(X_test, step_size=1e-9, upper_bound=1.0, max_iter=max_iter)
		rob_metrics_train['local_eps'] += nb.predict_local_epsilon_rob(X_train, step_size=1e-9, upper_bound=1.0, max_iter=max_iter)


	if 'total_var' in metrics_to_calc:
		rob_metrics['rob_loc_TV'] += nb.predict_local_TV_rob(X_test, max_iter=max_iter, min_rob=1e-10)
		rob_metrics_train['rob_loc_TV'] += nb.predict_local_TV_rob(X_train, max_iter=max_iter, min_rob=1e-10)

	if 'chi2' in metrics_to_calc:
		rob_metrics['rob_loc_chi2'] += nb.predict_local_chi2_rob(X_test, step_size=1e-9, upper_bound=1.0, max_iter=max_iter)
		rob_metrics_train['rob_loc_chi2'] += nb.predict_local_chi2_rob(X_train, step_size=1e-9, upper_bound=1.0, max_iter=max_iter)

	return rob_metrics, classification_results, y_pred, rob_metrics_train, classification_results_train, y_pred_train
