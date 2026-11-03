import numpy as np

from robustness_quantification.robustness.nbc_robustness import robustness_check_eps, robustness_check_TV, robustness_check_chi2


class NBC():
    """
        Class that defines and evaluates a Naive Bayes Classifier.

        Attributes
        ----------
        alpha: float
            The smoothing parameter for Laplace smoothing.
        class_counts: numpy array
            The counts of each class in the training data.
        classes: numpy array
            The unique classes in the training data.
        classes_num: numpy array
            The unique classes in the training data, represented as integers.
        feat_vals: list of numpy arrays
            The unique feature values for each feature in the training data.
        feat_val_dicts: list of dicts
            A list of dictionaries that map feature values to their corresponding indices for each feature.
        prior: numpy array
            The prior probabilities of each class.
        likelihood: list of lists of numpy arrays
            The likelihood of each feature value given each class.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.class_counts = None
        self.classes = None
        self.classes_num = None
        self.feat_vals = None
        self.feat_val_dicts = []
        self.prior = None
        self.likelihood = None


    def fit(self, X, y, N=100, feat_vals=None, classes=None):
        """
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.
        y : ndarray of shape (n_samples,)
            Class labels.
        """
        if classes is not None:
            self.classes = classes
        else:
            self.classes = np.unique(y)

        # set class/feature labels to 0, 1, 2, ...
        X_num = np.zeros(X.shape)

        if feat_vals is None:
            feat_vals = [np.unique(X[:, i]) for i in range(X.shape[1])]
        self.feat_vals = feat_vals

        for i in range(len(self.feat_vals)):
            X_num[:, i] = np.searchsorted(self.feat_vals[i], X[:, i])
            self.feat_val_dicts.append(dict(zip(self.feat_vals[i], np.arange(len(self.feat_vals[i])))))
        X_num = X_num.astype(int)

        self.classes_num = np.arange(len(self.classes))
        y_num = np.searchsorted(self.classes, y)

        self.prior = self.class_prior(y_num)
        self.likelihood = self.calculate_likelihood(X, y_num)


    def class_prior(self, y):
        """
        This function calculates the class prior as the relative frequencies of each class.
        ----------
        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Class labels. These labels should be numeric and go from 0 to len(self.classes)

        Returns
        -------
        prior : ndarray of shape (n_classes,)
            Class priors with additive (Laplace) smoothing.
        """

        # Returns an array with at index i the counts of class i
        self.class_counts = np.bincount(y)

        # if length of class_counts is not equal to the number of classes, add 0 counts for missing classes
        if len(self.class_counts) != len(self.classes):
            self.class_counts = np.append(self.class_counts, np.zeros(len(self.classes) - len(self.class_counts)))

        class_prior = (self.class_counts + self.alpha) / (len(y) + len(self.classes) * self.alpha)

        return class_prior


    def calculate_likelihood(self, X, y):
        """
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.
        y : ndarray of shape (n_samples,)
            Class labels.

        Returns
        -------
        likelihood : list of list of arrays (n_classes, n_features, n_feature_values (variable))
            Likelihood of the features.
        """

        likelihood = []

        for _, c in enumerate(self.classes_num):
            # all samples of class c
            X_class_c = X[y == c]
            class_count = len(X_class_c)
            class_likelihoods = []

            for j, f_vals in enumerate(self.feat_vals):
                feat_class_likelihoods = []

                for feature in f_vals:
                    feat_class_count = len(X_class_c[:,j][X_class_c[:,j] == feature])
                    feat_class_likel = (feat_class_count + self.alpha) / (class_count + len(f_vals) * self.alpha)

                    feat_class_likelihoods.append(feat_class_likel)

                class_likelihoods.append(np.array(feat_class_likelihoods))

            likelihood.append(class_likelihoods)
        return likelihood


    def predict_posterior(self, X):
        """
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Predicted log probability of each class per sample.
        """

        log_prior = np.log(self.prior)

        posterior = np.zeros((len(X), len(self.classes_num)))

        for sample_i in range(len(X)):
            for cl in range(len(self.classes_num)):
                posterior[sample_i][cl] += log_prior[cl]

                for feat_i in range(len(self.feat_vals)):
                    feat_val = self.feat_val_dicts[feat_i][X[sample_i][feat_i]]

                    posterior[sample_i][cl] += np.log(self.likelihood[cl][feat_i][feat_val])

        return np.array(posterior)


    def get_joint_distribution(self):
        num_classes = len(self.classes)
        num_feature_values = [len(self.feat_vals[i]) for i in range(len(self.feat_vals))]

        # return the joint distribution of the features and the classes
        joint = np.zeros((num_classes, np.prod(num_feature_values)))
        for C in range(num_classes):
            pc = self.prior[C]
            for i in range(np.prod(num_feature_values)):
                pcfi = pc
                i_orig = i
                for j in range(len(num_feature_values)):
                    pcfi *= self.likelihood[C][j][i % num_feature_values[j]]
                    i = i // num_feature_values[j]
                joint[C][i_orig] = pcfi

        # normalize the joint distribution
        joint = joint / np.sum(joint)
        return joint


    def predict(self, X):
        """
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted classes for X
        """
        proba = self.predict_posterior(X)
        return self.classes[np.argmax(proba, axis=1)]


    def predict_local_epsilon_rob(self, X, step_size=0.001, upper_bound=0.5, max_iter=60):
        """
        Predict the smallest epsilon per sample for which at least one *non-winning*
        class' (log) posterior score becomes >= the sample's current maximum (log) posterior.

        This version uses the bisection method (binary search), exploiting that the
        competing class score is monotone increasing in epsilon.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.
        step_size : float, default=0.001
            Resolution knob, matching the spirit of the original implementation.
            We stop when the bisection bracket is smaller than step_size/10.
        upper_bound : float, default=1.0
            Upper bound of the epsilon search interval.
        max_iter : int, default=60
            Maximum bisection iterations per sample.

        Returns
        -------
        epsilons : ndarray of shape (n_samples,)
            Predicted epsilon thresholds for X.
        """

        # Compute current (log) posteriors and identify winning class per sample
        pcf = self.predict_posterior(X)
        winner_k = np.argmax(pcf, axis=1)
        winner_score = np.max(pcf, axis=1)

        # Pre-pack priors and likelihoods in class order (same as pcf columns)
        classes = list(self.classes_num)  # already in same order as pcf columns

        prior = np.asarray([self.prior[c] for c in classes], dtype=np.float64)

        # likelihood: shape (K, n_features, n_values_per_feature)
        # Assumes likelihood[c][j][feature_num] is accessible as shown in original code.
        # not all features have the same number of values, so to make this into an array instead of a nested list, we will pad the likelihoods for features with fewer values with np.nan (these values should never be selected, so this shouldn't lead to errors).
        likelihood_padded = []
        max_n_values = max(len(f_vals) for f_vals in self.feat_vals)
        for c in classes:
            class_likelihoods = []
            for j in range(len(self.feat_vals)):
                feat_likel = self.likelihood[c][j]
                if len(feat_likel) < max_n_values:
                    feat_likel = np.pad(feat_likel, (0, max_n_values - len(feat_likel)), constant_values=np.nan)
                class_likelihoods.append(feat_likel)
            likelihood_padded.append(class_likelihoods)

        likelihood = np.asarray([likelihood_padded[c] for c in classes], dtype=np.float64)

        n_samples, n_features = X.shape
        feat_positions = np.arange(n_features)

        # Define search tolerance and safe upper bound (< 1) to avoid division by 0
        tol = step_size / 10.0
        min_eps = tol
        hi_cap = min(float(upper_bound), 1.0) - 1e-12
        hi_cap = max(hi_cap, 1e-12)

        epsilons = np.empty(n_samples, dtype=np.float64)

        # bisection to find smallest eps
        for i in range(n_samples):
            sample = X[i]

            feat_idx = np.fromiter(
                (self.feat_val_dicts[j][sample[j]] for j in range(n_features)),
                dtype=np.int64,
                count=n_features
            )

            like_vals = likelihood[:, feat_positions, feat_idx]


            params_mat = np.concatenate([prior[:, None], like_vals], axis=1)

            win_idx = winner_k[i]
            target = winner_score[i]


            if robustness_check_eps(0.0, params_mat, win_idx, target):
                epsilons[i] = min_eps / 10.0
                continue

            if not robustness_check_eps(hi_cap, params_mat, win_idx, target):
                # If even near 1.0 no competitor reaches target, return capped high bound.
                epsilons[i] = max(hi_cap, min_eps)
                continue


            lo, hi = 0.0, hi_cap

            for _ in range(max_iter):
                mid = (lo + hi) * 0.5
                if robustness_check_eps(mid, params_mat, win_idx, target):
                    hi = mid
                else:
                    lo = mid

                if (hi - lo) <= tol:
                    break

            # Prevent epsilon from being exactly 0
            epsilons[i] = max(hi, min_eps)

        return epsilons


    def predict_local_TV_rob(self, X, max_iter=25, min_rob=1e-8):
        """
        Predict the smallest delta of a TV-ball per sample for which at least one *non-winning*
        class' (log) posterior score becomes >= the sample's current maximum (log) posterior.

        This method uses the bisection method (binary search), exploiting that the
        competing class score is monotone increasing in epsilon.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.
        step_size : float, default=0.001
            Resolution knob, matching the spirit of the original implementation.
            We stop when the bisection bracket is smaller than step_size/10.
        upper_bound : float, default=1.0
            Upper bound of the epsilon search interval. Must be <= 1.0.
        max_iter : int, default=60
            Maximum bisection iterations per sample.


        lambda = upper probs of features probs conditional on not_winning class
            upper prob = min{1, P(fi|c) + delta}
        mu = lower probs of features probs conditional on predicted class
            lower prob analogous

        NOT robust if at least for one non_winning class:
            delta > (P(c^)*mu - P(c)*lambda) / (lambda - mu)

        Returns
        -------
        robs : ndarray of shape (n_samples,)
            Predicted robustness thresholds for X.
        """

        # Compute current (log) posteriors and identify winning class per sample
        pcf = self.predict_posterior(X)
        winner_k = np.argmax(pcf, axis=1)
        winner_score = np.max(pcf, axis=1)

        # Pre-pack priors and likelihoods in class order (same as pcf columns)
        classes = list(self.classes_num)  # already in same order as pcf columns

        prior = np.asarray([self.prior[c] for c in classes], dtype=np.float64)

        # likelihood: shape (K, n_features, n_values_per_feature)
        # Assumes likelihood[c][j][feature_num] is accessible as shown in original code.
        # not all features have the same number of values, so to make this into an array instead of a nested list, we will pad the likelihoods for features with fewer values with np.nan (these values should never be selected, so this shouldn't lead to errors).
        likelihood_padded = []
        max_n_values = max(len(f_vals) for f_vals in self.feat_vals)
        for c in classes:
            class_likelihoods = []
            for j in range(len(self.feat_vals)):
                feat_likel = self.likelihood[c][j]
                if len(feat_likel) < max_n_values:
                    feat_likel = np.pad(feat_likel, (0, max_n_values - len(feat_likel)), constant_values=np.nan)
                class_likelihoods.append(feat_likel)
            likelihood_padded.append(class_likelihoods)

        likelihood = np.asarray([likelihood_padded[c] for c in classes], dtype=np.float64)

        n_samples, n_features = X.shape
        feat_positions = np.arange(n_features)


        robs = np.empty(n_samples, dtype=np.float64)


        # bisection to find smallest rob where predicate becomes True
        for i in range(n_samples):
            sample = X[i]

            feat_idx = np.fromiter(
                (self.feat_val_dicts[j][sample[j]] for j in range(n_features)),
                dtype=np.int64,
                count=n_features
            )

            likel_vals = likelihood[:, feat_positions, feat_idx]

            win_idx = winner_k[i]

            hi_cap = np.min(likel_vals[win_idx])

            if robustness_check_TV(0.0, prior, likel_vals, win_idx):
                robs[i] = min_rob / 10.0
                continue

            if not robustness_check_TV(hi_cap, prior, likel_vals, win_idx):
                # If even near 1.0 no competitor reaches target, return capped high bound.
                robs[i] = max(hi_cap, min_rob)
                continue


            lo, hi = 0.0, hi_cap

            for _ in range(max_iter):
                mid = (lo + hi) * 0.5
                if robustness_check_TV(mid, prior, likel_vals, win_idx):
                    hi = mid
                else:
                    lo = mid

                if (hi - lo) <= min_rob:
                    break

            # Prevent robustness from being exactly 0
            robs[i] = max(hi, min_rob)

        return robs



    def predict_local_chi2_rob(self, X, step_size=0.00001, upper_bound=1.0, max_iter=60):
        """
        Predict the smallest delta of a Chi-squared ball per sample for which at least one *non-winning*
        class' (log) posterior score becomes >= the sample's current maximum (log) posterior.

        This method uses the bisection method (binary search), exploiting that the
        competing class score is monotone increasing in epsilon.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input samples.
        step_size : float, default=0.001
            Resolution knob, matching the spirit of the original implementation.
            We stop when the bisection bracket is smaller than step_size/10.
        upper_bound : float, default=1.0
            Upper bound of the epsilon search interval.
        max_iter : int, default=60
            Maximum bisection iterations per sample.


        lambda = upper probs of features probs conditional on not_winning class
            upper prob = min{1, P(fi|c) + delta}
        mu = lower probs of features probs conditional on predicted class
            lower prob analogous

        NOT robust if at least for one non_winning class:
            delta > (P(c^)*mu - P(c)*lambda) / (lambda - mu)

        Returns
        -------
        robs : ndarray of shape (n_samples,)
            Predicted robustness thresholds for X.
        """

        # Compute current (log) posteriors and identify winning class per sample
        pcf = self.predict_posterior(X)
        winner_k = np.argmax(pcf, axis=1)
        winner_score = np.max(pcf, axis=1)

        # Pre-pack priors and likelihoods in class order (same as pcf columns)
        classes = list(self.classes_num)  # already in same order as pcf columns

        prior = np.asarray([self.prior[c] for c in classes], dtype=np.float64)

        # likelihood: shape (K, n_features, n_values_per_feature)
        # Assumes likelihood[c][j][feature_num] is accessible as shown in original code.
        # not all features have the same number of values, so to make this into an array instead of a nested list, we will pad the likelihoods for features with fewer values with np.nan (these values should never be selected, so this shouldn't lead to errors).
        likelihood_padded = []
        max_n_values = max(len(f_vals) for f_vals in self.feat_vals)
        for c in classes:
            class_likelihoods = []
            for j in range(len(self.feat_vals)):
                feat_likel = self.likelihood[c][j]
                if len(feat_likel) < max_n_values:
                    feat_likel = np.pad(feat_likel, (0, max_n_values - len(feat_likel)), constant_values=np.nan)
                class_likelihoods.append(feat_likel)
            likelihood_padded.append(class_likelihoods)

        likelihood = np.asarray([likelihood_padded[c] for c in classes], dtype=np.float64)

        n_samples, n_features = X.shape
        feat_positions = np.arange(n_features)

        # Define search tolerance and safe upper bound (< 1) to avoid division by 0
        tol = step_size / 10.0
        min_rob = tol
        hi_cap = min(float(upper_bound), 1.0) - 1e-12
        hi_cap = max(hi_cap, 1e-12)

        robs = np.empty(n_samples, dtype=np.float64)


        # Bisection to find smallest rob where predicate becomes True
        for i in range(n_samples):
            sample = X[i]


            feat_idx = np.fromiter(
                (self.feat_val_dicts[j][sample[j]] for j in range(n_features)),
                dtype=np.int64,
                count=n_features
            )


            likel_vals = likelihood[:, feat_positions, feat_idx]

            win_idx = winner_k[i]

            if robustness_check_chi2(0.0, prior, likel_vals, win_idx):
                robs[i] = min_rob / 10.0
                continue


            if not robustness_check_chi2(hi_cap, prior, likel_vals, win_idx):
                # If even near 1.0 no competitor reaches target, return capped high bound.
                robs[i] = max(hi_cap, min_rob)
                continue

            lo, hi = 0.0, hi_cap

            for _ in range(max_iter):
                mid = (lo + hi) * 0.5
                if robustness_check_chi2(mid, prior, likel_vals, win_idx):
                    hi = mid
                else:
                    lo = mid

                if (hi - lo) <= tol:
                    break

            robs[i] = max(hi, min_rob)

        return robs


###########################
## NBC utility functions ##
###########################

def calculate_max_prob(pred_prob):
    # calculate P(f) for each data point, the normalizing constant
    Pf = np.sum(np.exp(pred_prob), axis=1)

    # calculate the predicted, normalized probabilities for each class
    Post = np.exp(pred_prob)/Pf[:,None]

    # get the highest probability for each data point
    max_prob = np.max(Post, axis=1)
    return max_prob


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


def calculate_entropy(pred_prob):
    # calculate P(f) for each data point, the normalizing constant
    Pf = np.sum(np.exp(pred_prob), axis=1)

    # calculate the predicted, normalized probabilities for each class
    Post = np.exp(pred_prob)/Pf[:,None]

    # calculate the entropy for each data point
    entropies = -np.sum(Post * np.log2(Post), axis=1)
    return entropies


def bootstrap(X, y, n_samples):
    indices = np.random.choice(X.shape[0], n_samples, replace=True)
    return X[indices], y[indices]


def calculate_epistimic_uncertainty_bootstrapped(M_models, X_train, y_train, X_test, alpha, feat_vals=None):
    sum_pred_prob = np.zeros((len(X_test), len(np.unique(y_train))))

    entropies = np.zeros((len(X_test), M_models))

    for i in range(M_models):
        X_boot, y_boot = bootstrap(X_train, y_train, n_samples=len(X_train))

        nb_boot = NBC(alpha=alpha)
        nb_boot.fit(X_boot, y_boot, feat_vals=feat_vals, classes=np.unique(y_train))

        pred_prob = nb_boot.predict_posterior(X_test)

        norm = np.sum(np.exp(pred_prob), axis=1)
        norm_pred_prob = np.exp(pred_prob) / norm[:,None]
        sum_pred_prob += norm_pred_prob

        entropy = np.sum(norm_pred_prob * np.log2(norm_pred_prob), axis=1)
        entropies[:, i] = entropy

    avg_entropy = np.mean(entropies, axis=1)
    max_entropy = np.max(entropies, axis=1)
    min_entropy = np.min(entropies, axis=1)
    minmax_diff = max_entropy - min_entropy

    avg_pred_prob = sum_pred_prob / M_models
    entropy_avg = np.sum(avg_pred_prob * np.log2(avg_pred_prob), axis=1)

    return avg_entropy, entropy_avg, minmax_diff