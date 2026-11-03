import numpy as np

###########################
##### SPLIT FUNCTIONS #####
###########################

class Split:
    def __init__(self):
        """
            Abstract class for storing the relevant information of a split.

		Attributes
		----------
        score: float
            The score of the split.
        split_feature: int
            The index of the feature that is split on.
        threshold: numpy array
            The feature values for which the left child is chosen.
        left_ids, right_ids: numpy arrays
            The indices of the samples that go to the left and right child, respectively.
        left_counts, right_counts: numpy arrays
            The counts of the class variable for the left and right child, respectively.

        """
        self.score = -np.inf
        self.split_feature = -1
        self.threshold = np.zeros(1, dtype=np.float64) -np.inf

        self.left_ids = np.empty(0, dtype=np.int64)
        self.right_ids = np.empty(0, dtype=np.int64)

        self.left_counts = np.empty(0, dtype=np.int64)
        self.right_counts = np.empty(0, dtype=np.int64)



def categorical_split(x, y, idx, split_feature, total_counts, tree):
    """
        Looks for the best split for categorical variables. It differs from
        a numerical split in that it is not sufficient to divide values above
        and below are threshold because the variable is not necessarily ordinal.
        Because of that all possible subsets of the categories need to be
        considered for a split.

        If the target variable is binary, we can do better by ordering the
        categories in `x` according to the proportion falling in class 1.
        Then we split `x` as if it were an ordered predictor.
        - from Elements of Statistical Learning p.310

        Parameters
        ----------
        x: numpy
            One dimensional array containing the realisations of a categorical
            variable we want to split on. Assumed without missing values.
        y: numpy
            The class variable corresponding to the observations in x.
        idx: numpy
            The indices of the examples in x and y w.r.t. to the entire dataset.
        split_feature: int
            The index (column) of the feature to split on.
        total_counts: array of ints
            The counts of the target variable over all samples.
        tree: Tree object
            Only for assessing tree parameters, namely min_samples_leaf and
            number of classes.

        Returns
        -------
        best_split: Split object
    """
    best_split = Split()
    # Create an array with counts per category
    counts = np.zeros((tree.n_feat_vals[split_feature], tree.n_classes), dtype=np.int64)
    x_int = np.asarray(x, dtype=np.int64)
    for i in range(len(x)):
        counts[x_int[i], y[i]] += 1

    if tree.n_classes == 2:
        categories = np.argsort(counts[:, 1].ravel())
        n_splits = tree.n_feat_vals[split_feature]
    else:
        categories = np.arange(tree.n_feat_vals[split_feature])
        n_splits = int(2**(categories.shape[0]-1))

    for i in range(1, n_splits):
        if tree.n_classes == 2:
            split = np.array([categories[j] for j in range(categories.shape[0]) if j < i])
        else:
            split = np.array([categories[j] for j in range(categories.shape[0]) if (i & (1 << j))])

        left_size = counts[split].sum()
        right_size = y.size-left_size
        if (left_size < tree.min_samples_leaf) or (right_size < tree.min_samples_leaf):
            score = -np.inf
        else:
            left_counts = counts[split].sum(axis=0)
            right_counts = total_counts - left_counts
            score = gain(left_counts, right_counts, tree.n_classes, tree.impurity_measure)

        if score > best_split.score:
            left_mask = np.isin(x, split)
            best_split.score = score
            best_split.threshold = split.astype(np.float64)
            best_split.left_ids = idx[left_mask]
            best_split.right_ids = idx[~left_mask]
            best_split.left_counts = left_counts
            best_split.right_counts = right_counts
    return best_split


def find_best_split(node, tree, random_state):
    """
        Looks for the best split using the data reaching node.

        Parameters
        ----------
        node: Node object (nodes.py)
            The node for which we are looking for the best split.
        tree: Tree object (trees.py)
            The tree to which the node belongs.

        Returns
        -------
        best_split: Split object containing the info of the best split.
    """
    if (node.idx.shape[0] < tree.min_samples_split) or (purity(node.counts) >= 0.99) or (node.depth >= tree.max_depth):
        return None

    best_score = -1e6
    best_split = None
    np.random.seed(random_state)
    possible_split_features = np.random.choice(np.arange(tree.X.shape[1]), tree.max_features, replace=False)
    X = tree.X[node.idx, :]
    y = tree.y[node.idx]

    for i in range(tree.max_features):
        split_feature = possible_split_features[i]
        x = X[:, split_feature]

        split = categorical_split(x, y, node.idx, split_feature, node.counts, tree)

        if split.score > best_score:
            best_score = split.score
            best_split = split
            best_split.split_feature = split_feature

    return best_split

#############################

def resample_stratified(x, y, n_classes, random_state=None):
    """
    Resampling (bootstrapping) of x stratified according to y.

    Parameters
    ----------
    x: numpy array
        Typically a matrix nxm with n samples and m variables.
    y: numpy array (1D)
        The class variable we are stratifying against.
    n_classes: int
        The number of classes in y.

    Returns
    -------
    x, y: numpy arrays
        In-bag samples.
    """
    if random_state is not None:
        np.random.seed(random_state)
    idx = np.arange(x.shape[0], dtype=np.int64)
    counts = np.bincount(y, minlength=n_classes)
    selected_idx = np.empty(0, dtype=np.int64)
    for i in range(n_classes):
        s = np.random.choice(idx[y==i], counts[i], replace=True)
        selected_idx = np.concatenate((selected_idx, s))
    return x[selected_idx, :], y[selected_idx], selected_idx


def bincount(data, n):
    counts = np.zeros(n, dtype=np.int64)
    for j in range(n):
        counts[j] = np.sum(data==j)
    return counts


#############################
##### IMPURITY MEASURES #####
#############################

def gini(left_counts, right_counts, n):
    total_left = 0
    total_right = 0
    gini_left = 0
    gini_right = 0
    for i in range(n):
        gini_left -= left_counts[i]**2
        gini_right -= right_counts[i]**2
        total_left += left_counts[i]
        total_right += right_counts[i]

    gini_left = total_left - gini_left/total_left
    gini_right = total_right - gini_right/total_right
    return gini_left + gini_right


def entropy(counts):
    """
        Computes the entropy in the distribution of classes.
        Returns -1*entropy as we want to maximize instead of minimize the score.
    """
    p = counts/np.sum(counts) + 1e-12
    return np.sum(p*np.log2(p))


def purity(counts):
    """
        Purity is a measure defined by the ratio between the number of counts
        of the majority class over the total number of instances.
        The larger the purity, the better the cluster.
    """
    if counts.sum() == 0:
        return 1
    return np.max(counts)/np.sum(counts)


def gain(left_counts, right_counts, n, impurity_measure):
    """
        Computes the gain of a split.

        Parameters
        ----------
        left_counts, right_counts: numpy
            The counts in each side of the split.
        impurity_measure: string
            The type of impurity of measure to use.
    """
    if impurity_measure == 'gini':
        return gini(left_counts, right_counts, n)
    ratio = np.sum(left_counts)/np.sum(left_counts + right_counts)
    if impurity_measure == 'purity':
        return ratio*purity(left_counts) + (1-ratio)*purity(right_counts)
    elif impurity_measure == 'entropy':
        return ratio*entropy(left_counts) + (1-ratio)*entropy(right_counts)
    else:
        return -np.inf



#############################
##  Perturbation functions ##
#############################

def perturb_weights(weight, perturbation_size, perturbation_type, minimize=True):
    """
        Perturbs the weights of a node according to a certain perturbation size and type.

        Parameters
        ----------
        weight: numpy array
            The weight of the node.
        perturbation_size: float
            The size of the perturbation.
        perturbation_type: str
            The type of perturbation. Can be 'econt', 'chi2' or 'total_var'.
        minimize: bool
            Whether to minimize or maximize the perturbed weights.

        Returns
        -------
        perturbed_weight: numpy array
            The perturbed weights of the node.
    """
    if perturbation_type == 'econt':
        perturbed_weight = weight * (1 - perturbation_size)
        if not minimize:
            perturbed_weight += perturbation_size

    elif perturbation_type == 'total_var':
        if minimize:
            perturbed_weight = max(weight - perturbation_size, 0)
        else:
            perturbed_weight = min(weight + perturbation_size, 1)

    elif perturbation_type == 'chi2':
        if minimize:
            perturbed_weight = max(weight - np.sqrt(perturbation_size*weight*(1-weight)), 0)
        else:
            perturbed_weight = min(weight + np.sqrt(perturbation_size*weight*(1-weight)), 1)

    else:
        raise ValueError("Perturbation type must be 'econt', 'chi2' or 'total_var'.")

    return perturbed_weight