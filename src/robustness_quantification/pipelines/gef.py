import numpy as np

from GeF import GeF, compute_robustness, calc_information_theoretic_uncertainty, calculate_entropy, calculate_precise_diff, get_rob_glob


def standard_experiment_GeFs(X_train, y_train, X_test, y_test, C, F, metrics_to_calc, n_estimators=40, N_bootstraps=10, precision_bisearch=1e-10, max_iter=100):
	"""
		This experiment fits a GeF to a dataset, predicts the classes of a test set and the corresponding probabilities, and then computes all the reliability metrics on the test set.

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
				n_estimators: int
					Number of trees in the GeF.
				N_bootstraps: int
					Number of bootstraps to use for calculating the ensemble uncertainty metrics.
				precision_bisearch: float
					Precision for the bisection search when computing the robustness metrics.
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

	do_total_variation = False
	do_econt = False
	do_chi2 = False

	# initialize the arrays to store the results
	rob_metrics = {}
	for _, metric in enumerate(metrics_to_calc):
		rob_metrics[metric] = np.zeros(N_test)
		if metric == 'total_var':
			do_total_variation = True
		if metric == 'local_eps':
			do_econt = True
		if metric == 'chi2':
			do_chi2 = True


	classification_results = np.zeros(N_test)
	y_pred = np.zeros(N_test)

	# this array contains the possible values for each feature
	ncat = np.array(F + [C])

	print(X_train.shape, y_train.shape)
	gef = GeF(n_feat_vals=F, n_classes=C, n_estimators=n_estimators, random_state=42)
	gef.fit(X_train, y_train)


    # Predict and compute the robustness of each test sample and save to a dictionary
	if do_econt:
		y_pred, pred_prob, rob_econt = compute_robustness(gef, X_test, perturbation_type='econt', max_iter=max_iter, precision_bisearch=precision_bisearch)

	if do_total_variation:
		y_pred, pred_prob, rob_totalvar = compute_robustness(gef, X_test, perturbation_type='total_var', max_iter=max_iter, precision_bisearch=precision_bisearch)

	if do_chi2:
		y_pred, pred_prob, rob_chi2 = compute_robustness(gef, X_test, perturbation_type='chi2', max_iter=max_iter, precision_bisearch=precision_bisearch)


	rob_glob, margin_of_confidence, max_prob = get_rob_glob(pred_prob)

	entropy = calculate_entropy(pred_prob)

	total_unc, aleatoric_unc, epistemic_unc = calc_information_theoretic_uncertainty(N_bootstraps=N_bootstraps, X_train=X_train, y_train=y_train, X_test=X_test, ncat=ncat, n_estimators=n_estimators)

	classification_results += (y_test == y_pred).astype(float)


	### calculate the reliability metrics

	## uncertainty metrics:
	if 'max_prob' in metrics_to_calc:
		# probability of the predicted class (highest one) for each data point
		rob_metrics['max_prob'] += max_prob


	if 'entropies' in metrics_to_calc:
		# entropy of the predicted probabilities for each data point
		rob_metrics['entropies'] -= entropy


	# ensemble uncertainty metrics
	if 'aleatoric' in metrics_to_calc or 'epistemic' in metrics_to_calc or 'total' in metrics_to_calc:
		if 'aleatoric' in metrics_to_calc:
			rob_metrics['aleatoric'] += aleatoric_unc
		if 'total' in metrics_to_calc:
			rob_metrics['total'] += total_unc
		if 'epistemic' in metrics_to_calc:
			rob_metrics['epistemic'] += epistemic_unc


	if 'margin_of_confidence' in metrics_to_calc or 'global_eps' in metrics_to_calc:

		if 'margin_of_confidence' in metrics_to_calc:
			rob_metrics['margin_of_confidence'] += margin_of_confidence

		# global epsilon (robustness metric), is a scaled version of the deltas
		if 'global_eps' in metrics_to_calc:
			rob_metrics['global_eps'] += rob_glob

	## robustness metrics:
	if 'local_eps' in metrics_to_calc:
		rob_metrics['local_eps'] += rob_econt

	if 'total_var' in metrics_to_calc:
		rob_metrics['total_var'] += rob_totalvar

	if 'chi2' in metrics_to_calc:
		rob_metrics['chi2'] += rob_chi2

	return rob_metrics, classification_results, y_pred


def standard_experiment_on_train_GeFs(X_train, y_train, X_test, y_test, C, F, metrics_to_calc, n_estimators=40, N_bootstraps=10, precision_bisearch=1e-10, max_iter=100):
	"""
		This experiment fits a GeF to a dataset, predicts the classes of a test set and the corresponding probabilities, and then computes all the reliability metrics on the test set.
		It does the same on the training set as well, to tune hyperparameters later on.

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
				n_estimators: int
					Number of trees in the GeF.
				N_bootstraps: int
					Number of bootstraps to use for calculating the ensemble uncertainty metrics.
				precision_bisearch: float
					Precision for the bisection search when computing the robustness metrics.
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
	N_train = X_train.shape[0]

	do_total_variation = False
	do_econt = False
	do_chi2 = False

	# initialize the arrays to store the results
	rob_metrics = {}
	rob_metrics_train = {}
	for _, metric in enumerate(metrics_to_calc):
		rob_metrics[metric] = np.zeros(N_test)
		rob_metrics_train[metric] = np.zeros(N_train)
		if metric == 'total_var':
			do_total_variation = True
		if metric == 'local_eps':
			do_econt = True
		if metric == 'chi2':
			do_chi2 = True


	classification_results = np.zeros(N_test)
	y_pred = np.zeros(N_test)

	classification_results_train = np.zeros(N_train)
	y_pred_train = np.zeros(N_train)

	ncat = np.array(F + [C])
	print(X_train.shape, y_train.shape)
	gef = GeF(n_feat_vals=F, n_classes=C, n_estimators=n_estimators, random_state=42)
	gef.fit(X_train, y_train)


    # Predict and compute the robustness of each test sample and save to a dictionary
	if do_econt:
		y_pred, pred_prob, rob_econt = compute_robustness(gef, X_test, perturbation_type='econt', max_iter=max_iter, precision_bisearch=precision_bisearch)
		y_pred_train, pred_prob_train, rob_econt_train = compute_robustness(gef, X_train, perturbation_type='econt', max_iter=max_iter, precision_bisearch=precision_bisearch)

	if do_total_variation:
		y_pred, pred_prob, rob_totalvar = compute_robustness(gef, X_test, perturbation_type='total_var', max_iter=max_iter, precision_bisearch=precision_bisearch)
		y_pred_train, pred_prob_train, rob_totalvar_train = compute_robustness(gef, X_train, perturbation_type='total_var', max_iter=max_iter, precision_bisearch=precision_bisearch)

	if do_chi2:
		y_pred, pred_prob, rob_chi2 = compute_robustness(gef, X_test, perturbation_type='chi2', max_iter=max_iter, precision_bisearch=precision_bisearch)
		y_pred_train, pred_prob_train, rob_chi2_train = compute_robustness(gef, X_train, perturbation_type='chi2', max_iter=max_iter, precision_bisearch=precision_bisearch)


	rob_glob, margin_of_confidence, max_prob = get_rob_glob(pred_prob)
	rob_glob_train, margin_of_confidence_train, max_prob_train = get_rob_glob(pred_prob_train)

	entropy = calculate_entropy(pred_prob)
	entropy_train = calculate_entropy(pred_prob_train)

	total_unc, aleatoric_unc, epistemic_unc = calc_information_theoretic_uncertainty(N_bootstraps=N_bootstraps, X_train=X_train, y_train=y_train, X_test=X_test, ncat=ncat, n_estimators=n_estimators)
	total_unc_train, aleatoric_unc_train, epistemic_unc_train = calc_information_theoretic_uncertainty(N_bootstraps=N_bootstraps, X_train=X_train, y_train=y_train, X_test=X_train, ncat=ncat, n_estimators=n_estimators)

	classification_results += (y_test == y_pred).astype(float)
	classification_results_train += (y_train == y_pred_train).astype(float)


	### calculate the reliability metrics

	## uncertainty metrics:
	if 'max_prob' in metrics_to_calc:
		# probability of the predicted class (highest one) for each data point
		rob_metrics['max_prob'] += max_prob
		rob_metrics_train['max_prob'] += max_prob_train


	if 'entropies' in metrics_to_calc:
		# entropy of the predicted probabilities for each data point
		rob_metrics['entropies'] -= entropy
		rob_metrics_train['entropies'] -= entropy_train


	# ensemble uncertainty metrics
	if 'aleatoric' in metrics_to_calc or 'epistemic' in metrics_to_calc or 'total' in metrics_to_calc:
		if 'aleatoric' in metrics_to_calc:
			rob_metrics['aleatoric'] += aleatoric_unc
			rob_metrics_train['aleatoric'] += aleatoric_unc_train
		if 'total' in metrics_to_calc:
			rob_metrics['total'] += total_unc
			rob_metrics_train['total'] += total_unc_train
		if 'epistemic' in metrics_to_calc:
			rob_metrics['epistemic'] += epistemic_unc
			rob_metrics_train['epistemic'] += epistemic_unc_train


	if 'margin_of_confidence' in metrics_to_calc or 'global_eps' in metrics_to_calc:

		if 'margin_of_confidence' in metrics_to_calc:
			rob_metrics['margin_of_confidence'] += margin_of_confidence
			rob_metrics_train['margin_of_confidence'] += margin_of_confidence_train

		# global epsilon (robustness metric), is a scaled version of the deltas
		if 'global_eps' in metrics_to_calc:
			rob_metrics['global_eps'] += rob_glob
			rob_metrics_train['global_eps'] += rob_glob_train

	## robustness metrics:
	if 'local_eps' in metrics_to_calc:
		rob_metrics['local_eps'] += rob_econt
		rob_metrics_train['local_eps'] += rob_econt_train

	if 'total_var' in metrics_to_calc:
		rob_metrics['total_var'] += rob_totalvar
		rob_metrics_train['total_var'] += rob_totalvar_train

	if 'chi2' in metrics_to_calc:
		rob_metrics['chi2'] += rob_chi2
		rob_metrics_train['chi2'] += rob_chi2_train

	return rob_metrics, classification_results, y_pred, rob_metrics_train, classification_results_train, y_pred_train