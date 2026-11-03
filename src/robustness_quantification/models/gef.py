import numpy as np
import time
from math import floor

from robustness_quantification.models.gedt import build_gedt

"""
REMARKS:
- LICENSE should be added to the repository -> copy the license from the original GeFs repository.
- Multithreading was not used before, so we removed it for simplicity. It can be added later if needed.
- In the code it is assumed that classes are integers from 0 to ... Make sure that this is checked before fitting the model.
- when fitting Tree, bootstrapping is used. Necessary? Effects?
- surrogate splits are removed (this a way of dealing with missing values)
"""

class GeF:
	"""
		Class that defines and evaluates a Generative Forest (GeF).

		Inspiration is taken from the repository https://github.com/AlCorreia/GeFs of Alvaro Correia, which implements the original GeF algorithm (Correia et al., 2020).
		In the original code, a Random Forest is first fit and then changed to a (general) PC.
		In this implementation, the GeF is learned directly from data without any generalities for other types of PCs for the sake of simplicity, readability, and efficiency.

		Attributes
		----------
		root: Node object
			The root node of the GeF.
		n_nodes: int
			The number of nodes in the GeF.
		scope: numpy array
			The scope of the GeF, i.e., the variables that are used in the GeF. The last variable is the class variable.
		n_feat_vals: numpy array
			The number of features values/categories/classes of each variable. One for continuous variables.
		n_classes: int
			The number of classes in the data.
		n_estimators: int
			The number of trees in the GeF.
		impurity_measure: str
			The impurity measure to use for splitting nodes. Can be 'gini', 'entropy' or 'purity'.
		min_samples_split: int
			The minimum number of samples required to split an internal node.
		min_samples_leaf: int
			The minimum number of samples allowed in a leaf node.
		max_features: int
			The maximum number of features to consider when looking for the best split.
		max_depth: int
			The maximum depth of the tree.
		random_state: int
			The seed for the random number generator.
		learner: str
			Defines the learning method to use for learning the distributions of the leaf nodes. Currently, only LearnSPN or fully-factorised leaves are supported.
			Currently, only LearnSPN (Gens and Domingos, 2013).
	"""

	def __init__(self, n_feat_vals, n_classes=None, n_estimators=100, impurity_measure='gini', min_samples_split=2, min_samples_leaf=1, max_features=None, max_depth=1e6, bootstrap=True, random_state=None):

		self.root = None
		self.n_nodes = 0
		self.scope = None

		self.n_feat_vals = np.asarray(n_feat_vals, dtype=np.int64) #
		self.n_classes = n_classes #
		self.ncat = np.concatenate((self.n_feat_vals, np.asarray([self.n_classes], dtype=np.int64)))
		self.n_estimators = n_estimators #
		self.impurity_measure = impurity_measure #
		self.min_samples_split = min_samples_split #
		self.min_samples_leaf = min_samples_leaf #
		self.max_features = max_features #
		self.max_depth = max_depth #
		self.bootstrap = bootstrap #

		self.estimators = None

		if random_state is None:
			random_state = int(time.time())
		self.random_state = random_state #


	def fit(self, X, y, smoothing=1e-6, max_features=None):
		""" Fits the GeF to (X, Y). """
		y = y.astype(np.int64)
		if self.n_classes is None:
			self.n_classes = len(np.unique(y))

		if max_features is not None:
			self.max_features = max_features
		elif self.max_features is None:
			self.max_features = max(floor(X.shape[1]/3), 1)
			# self.max_features = X.shape[1]

		self.scope = np.arange(X.shape[1]+1, dtype=np.int64)
		# self.root = TreeNode(0, self.scope, n=1, root_node=True)

		self.estimators = []

		for _ in range(self.n_estimators):
			gedt = build_gedt(X, y, self.bootstrap,
									self.n_feat_vals, self.n_classes,
									self.impurity_measure,
									self.min_samples_split, self.min_samples_leaf,
									self.max_features, self.max_depth,
									self.random_state, smoothing)

			self.estimators.append(gedt)
			# self.root.add_child(gedt)
			self.n_nodes += gedt.n_nodes

	def predict(self, X):
		"""
		Classifies samples X by averaging the joint distributions of each tree.

		The probabilities are averaged in log-space using a numerically
		stable log-sum-exp calculation.
		"""
		logprobs = np.zeros(shape=(X.shape[0], self.n_estimators, self.n_classes))

		for i, estimator in enumerate(self.estimators):
			logprobs[:, i, :] = estimator.get_logprobs(X)

		# Calculate: log(mean(exp(logprobs), axis=1))
		largest = np.max(logprobs, axis=1, keepdims=True)

		logprobs_avg = (np.log(np.sum(np.exp(logprobs - largest),axis=1))
							+ largest.squeeze(axis=1) - np.log(self.n_estimators))

		return np.argmax(logprobs_avg, axis=1)

	def predict_probs(self, X):
		"""
			Calculates the probabilities of samples X by averaging the joint distributions of each tree.
			For this we first compute the joint distribution of each tree and then average them.
		"""
		logprobs = np.zeros(shape=(X.shape[0], self.n_estimators, self.n_classes))

		for i, estimator in enumerate(self.estimators):
			logprobs[:, i, :] = estimator.get_logprobs(X)

		# Calculate: mean(exp(logprobs), axis=1)
		largest = np.max(logprobs, axis=1, keepdims=True)

		logprobs_avg = (np.log(np.sum(np.exp(logprobs - largest),axis=1))
							+ largest.squeeze(axis=1) - np.log(self.n_estimators))

		return np.exp(logprobs_avg)

	def predict_logprobs(self, X):
		"""
			Calculates the log probabilities of  X by averaging the joint distributions of each tree.
			For this we first compute the joint distribution of each tree and then average them.
		"""
		logprobs = np.zeros(shape=(X.shape[0], self.n_estimators, self.n_classes))

		for i, estimator in enumerate(self.estimators):
			logprobs[:, i, :] = estimator.get_logprobs(X)

		# Calculate: log(mean(exp(logprobs), axis=1))
		largest = np.max(logprobs, axis=1, keepdims=True)

		logprobs_avg = (np.log(np.sum(np.exp(logprobs - largest),axis=1))
							+ largest.squeeze(axis=1) - np.log(self.n_estimators))

		return logprobs_avg

	def predict_logprobs_unaveraged(self, X):
		"""
			Calculates the log probabilities of samples X for the joint distributions of each tree for all estimators in the forest.
			For this we first compute the joint distribution of each tree.
			This is used for the perturbed evaluation of the GeF.
		"""
		logprobs = np.zeros(shape=(X.shape[0], self.n_estimators, self.n_classes))

		for i, estimator in enumerate(self.estimators):
			logprobs[:, i, :] =  estimator.get_logprobs(X)

		return logprobs

###########################
##  Robustness functions ##
###########################


def calculate_perturbed_gef(gef, x, predicted_class, predicted_classes_estimators, pert_size, perturbation_type):
	"""
		Computes the perturbed (log) probabilities of a given input x, given a perturbation size and the GeF.

		Parameters
		----------
		gef : GeF
			The GeF itself.
		x : np.ndarray
			The input data point for which to compute the perturbed probabilities.
		predicted_class : int
			The class predicted by the GeF for the input x.
		predicted_classes_estimators : np.ndarray
			The classes predicted by each estimator in the GeF for the input x.
		pert_size : float
			The size of the perturbation to apply.
		perturbation_type : str
			The type of perturbation to apply.
	"""
	logprobs = np.zeros(shape=(gef.n_estimators, gef.n_classes))
	for i, estimator in enumerate(gef.estimators):
		minimize = True
		if predicted_classes_estimators[i] != predicted_class:
			minimize = False
		pert_logprobs = estimator.get_perturbed_logprobs(x, pert_size, perturbation_type, minimize)

		logprobs[i] = pert_logprobs

	logprobs_avg = np.zeros(gef.n_classes)

	for j in range(gef.n_classes):
		class_logprobs = logprobs[:, j]

		largest = np.max(class_logprobs)

		if largest == -np.inf:
			logprobs_avg[j] = -np.inf
		else:
			result = np.sum(np.exp(class_logprobs - largest))
			logprobs_avg[j] = np.log(result) + largest - np.log(gef.n_estimators)

	return logprobs_avg

def compute_robustness_instance(gef, x, predicted_class, predicted_classes_estimators, perturbation_type='econt', precision_bisearch=1e-3, max_iter=100):
	"""
		Compute the robustness of the GeF` for a single instance `x`.
		This is done by doing a binary search over the perturbation size to find the largest perturbation that does not change the predicted class.

		Parameters
		----------
		gef: GeF object
			The GeF itself.
		x: np.ndarray
			Test instance m variables/features.
		predicted_class: int
			The predicted class for the instance `x`.
		predicted_classes_estimators: np.ndarray
			The predicted classes for the instance `x` across all estimators.
		perturbation_type: str
			Type of perturbation to apply. Possible values are 'econt', 'chi2', 'totalvar'; default is 'econt'.
		precision_bisearch: float
			Precision of the binary search. The search will stop when the difference between the lower and upper bounds is less than this value.
		max_iter: int
			Maximum number of iterations for the binary search. Default is 100.

		Returns
		-------
		robustness_score: float
			Robustness score for the instance `x`.
	"""
	lower, upper = 0.0, 1.0
	if precision_bisearch <= 0 or precision_bisearch >= 1:
		print("precision_bisearch must be positive. Continued with default value 1e-3.")
		precision_bisearch = 1e-3

	if perturbation_type == 'chi2' or perturbation_type == 'total_var':
		# for the chi2 perturbation, there is actually no upper bound, so we try some large values to check whether we can find a suitable upper bound. Also, the values could be both very large and very small. If we cannot find a suitable upper bound, we return np.inf as the robustness score.
		possible_upper_bounds = [2**(-30), 2**(-20), 2**(-10), 1.0, 2**10, 2**20, 2**30]
		for i, possible_upper in enumerate(possible_upper_bounds):
			pert_probs = calculate_perturbed_gef(gef, x, predicted_class, predicted_classes_estimators, possible_upper, perturbation_type)

			if np.argmax(pert_probs) != predicted_class:
				upper = possible_upper
				if i != 0:
					# take the previous value as the lower bound, since we know that the class did not change for that perturbation size
					lower = possible_upper_bounds[i-1]
				break
		else:
			# If we cannot find a suitable upper bound, we return np.inf as the robustness score.
			return np.inf

	it = 0
	while (lower + precision_bisearch < upper) and (it < max_iter):
		it += 1
		class_changed = False
		rob = (lower + upper) / 2
		pert_probs = calculate_perturbed_gef(gef, x, predicted_class, predicted_classes_estimators, rob, perturbation_type)

		if np.argmax(pert_probs) != predicted_class:
			class_changed = True

		if class_changed:
			upper = rob
		else:
			lower = rob

	return rob

def compute_robustness(gef, X_test, perturbation_type='econt', max_iter=100, precision_bisearch=1e-8):
    """
        Compute the robustness of the GeF rooted at `gef_root` for each instance in `X_test`.

        Parameters
        ----------
        gef: GeF object
            The GeF itself.
        X_test: numpy array n x m
            Test data with n samples and m variables.
        perturbation_type: str
            Type of perturbation to apply. Possible values are 'econt', 'chi2', 'totalvar'; default is 'econt'.

        Returns
        -------
        maxclass: numpy array of shape (n,)
			The predicted class for each instance in `X_test`.
        logprobs: numpy array of shape (n, C)
			The log probabilities for each class for each instance in `X_test`.
        robustness_scores: numpy array of shape (n,)
			Robustness scores for each instance in `X_test`.
    """
    robustness_scores = np.zeros(X_test.shape[0])
    logprobs_unav = gef.predict_logprobs_unaveraged(X_test)
    maxclasses = np.argmax(logprobs_unav, axis=2)
    logprobs_avg = gef.predict_logprobs(X_test)
    maxclass = np.argmax(logprobs_avg, axis=1)
    counter = 0
    for i in range(X_test.shape[0]):
        # if all estimators predict the same class, we can skip the robustness computation since no matter the perturbation, the class will not change.
        if np.all(maxclasses[i] == maxclass[i]):
            robustness_scores[i] = np.inf
            counter += 1
            continue

        robustness_scores[i] = compute_robustness_instance(gef, X_test[i, :], maxclass[i], maxclasses[i], perturbation_type=perturbation_type, max_iter=max_iter, precision_bisearch=precision_bisearch)

    return maxclass, logprobs_avg, robustness_scores



############################
##  Uncertainty functions ##
############################

def calculate_entropy(pred_prob):
    # calculate P(f) for each data point, the normalizing constant
    Pf = np.sum(np.exp(pred_prob), axis=1)

    # calculate the predicted, normalized probabilities for each class
    Post = np.exp(pred_prob)/Pf[:,None]

    # calculate the entropy for each data point
    entropies = -np.sum(Post * np.log2(Post), axis=1)
    return entropies


def get_rob_glob(logprobs):
	sorted_pred_prob = np.exp(np.sort(logprobs, axis=1))

	# get the highest and second highest probabilities for each data point
	max_prob = sorted_pred_prob[:,-1]
	second_prob = sorted_pred_prob[:,-2]

	diff_unnorm = max_prob - second_prob
	rob_glob = diff_unnorm/(1+diff_unnorm)

	# calculate P(f) for each data point, the normalizing constant
	Pf = np.sum(sorted_pred_prob, axis=1)

	# calculate the predicted, normalized probabilities for each class
	Post = sorted_pred_prob/Pf[:,None]

	# get the highest and second highest normalized probabilities for each data point
	max_prob_norm = Post[:,-1]
	second_prob_norm = Post[:,-2]

	margin_of_confidence = max_prob_norm - second_prob_norm

	return rob_glob, margin_of_confidence, max_prob_norm

def calculate_precise_diff(pred_prob, global_eps=True):
    sorted_pred_prob = np.exp(np.sort(pred_prob, axis=1))

    # get the highest and second highest probabilities for each data point
    max_prob = sorted_pred_prob[:,-1]
    second_prob = sorted_pred_prob[:,-2]

    # calculate P(f) for each data point, the normalizing constant
    Pf = np.sum(sorted_pred_prob, axis=1)

    # calculate the predicted, normalized probabilities for each class
    Post = sorted_pred_prob/Pf[:,None]

    # get the highest and second highest normalized probabilities for each data point
    max_prob_norm = Post[:,-1]
    second_prob_norm = Post[:,-2]

    diff_unnorm = max_prob - second_prob
    diff = max_prob_norm - second_prob_norm

    if global_eps:
        glob_eps = diff_unnorm/(1+diff_unnorm)
        return diff_unnorm, diff, glob_eps

    return diff_unnorm, diff


def bootstrap(X, y, n_samples):
    indices = np.random.choice(X.shape[0], n_samples, replace=True)
    return X[indices], y[indices]


def calc_information_theoretic_uncertainty(N_bootstraps, X_train, y_train, X_test, ncat, n_estimators):
    sum_pred_prob = np.zeros((len(X_test), int(ncat[-1])))

    entropies = np.zeros((len(X_test), N_bootstraps))

    for i in range(N_bootstraps):
        X_boot, y_boot = bootstrap(X_train, y_train, n_samples=len(X_train))

        gef = GeF(n_feat_vals=ncat[:-1], n_classes=ncat[-1], n_estimators=n_estimators)  # Train a Random Forest
        gef.fit(X_boot, y_boot)

        logprobs = gef.predict_logprobs(X_test)

        norm = np.sum(np.exp(logprobs), axis=1)
        norm_pred_prob = np.exp(logprobs) / norm[:,None]
        sum_pred_prob += norm_pred_prob

        entropy = np.sum(norm_pred_prob * np.log2(norm_pred_prob), axis=1)
        entropies[:, i] = entropy

    avg_entropy = np.mean(entropies, axis=1)

    avg_pred_prob = sum_pred_prob / N_bootstraps
    entropy_of_avg = np.sum(avg_pred_prob * np.log2(avg_pred_prob), axis=1)

    aleatoric_uncertainty = avg_entropy
    total_uncertainty = entropy_of_avg
    epistemic_uncertainty = total_uncertainty - aleatoric_uncertainty

    return total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty