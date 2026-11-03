import numpy as np

from GeDT_utils import find_best_split, resample_stratified, bincount, perturb_weights


def evaluate(node, X, n_classes=2):
    """
        Returns the corresponding class counts for each instance in X.

        Parameters
        ----------
        node: TreeNode object
            Typically the root node of the tree.
        X: numpy n x m
            Numpy array comprising n realisations (instances) of m variables.
        n_classes: int
            Number of classes in the data.

        Returns
        -------
        res: numpy n x c
            Numpy array comprising n realisations (instances) of c classes.
            Each row contains the class counts for the corresponding instance in X.
    """
    res = np.empty((X.shape[0], n_classes), dtype=np.float64)

    for i in range(X.shape[0]):
        res[i, :] = evaluate_instance(node, X[i, :])
    return res


def evaluate_instance(node, x):
    """
        Routes a single instance x and returns the corresponding class counts.

        Parameters
        ----------
        node: TreeNode object
            Typically the root node of the tree.
        x: numpy array of size m
            Numpy array comprising a single realisation of m variables.
        n_classes: int
            Number of classes in the data.

        Returns
        -------
        counts: numpy array of size n_classes
            Class counts for the instance x.
    """
    s = node.split
    if s is None:
        # Leaf Node
        return node.counts

    # Observed split variable
    go_left = np.any(np.isin(np.array(x[s.split_feature]), s.threshold))

    if go_left:
        return evaluate_instance(node.left_child, x)
    else:
        return evaluate_instance(node.right_child, x)


def evaluate_gedt(node, X, n_classes):
	"""
		Evaluates the GeDT rooted at `node` at data `X`.
		This function applies to any node in a GeDT and is called recursively.

		Parameters
		----------
		node: TreeNode object
			The root of the GeDT to evaluate.
		X: np.ndarray
			samples/data to evaluate the GeDT at.
		class_var: int
			Index of the class variable

        Returns
        -------
        logprs: np.ndarray
            Log-probabilities of the class variable for each sample in X.
	"""
	if node.isleaf:
		res = np.zeros((X.shape[0], n_classes))

		for i in range(X.shape[0]):
			class_logps = node.logps[-1]
			res[i, :] += class_logps

			for var in range(X.shape[1]):
				res[i, :] += node.logps[var][X[i, var]]

		return res

	else:
		logprs = np.zeros((X.shape[0], n_classes), dtype=np.float64)

		split_feature = node.split.split_feature
		feat_values_left = node.split.threshold

		lefts = np.where(np.isin(X[:, split_feature], feat_values_left))[0]
		if len(lefts) > 0:
			logprs[lefts, :] = evaluate_gedt(node.children[0], X[lefts, :], n_classes) + node.logw[0]

		rights = np.where(~np.isin(X[:, split_feature], feat_values_left))[0]
		if len(rights) > 0:
			logprs[rights, :] = evaluate_gedt(node.children[1], X[rights, :], n_classes) + node.logw[1]

		return logprs


def evaluate_gedt_perturbed(node, x, n_classes, perturbation_size, perturbation_type, minimize):
    """
        Evaluates the GeDT rooted at `node` for a single sample `x` for a certain perturbation size and type.
        This function applies to any node in a GeDT and is called recursively.

        Note: in a GeDT, for a certain instance x, the path from the root to a leaf is unique. Therefore, we only need to evaluate one path in the tree. All other paths will give a 0 probability for the instance x.
        So, also for the perturbed evaluation, we only consider the weights along this path, and perturb them.
        Also, if this particular GeDT predicts a different class than the predicted by the GeF, we need to maximize the perturbed weights along this path, otherwise, we minimize them.

        Parameters
        ----------
        node: TreeNode object
            The root of the GeDT to evaluate.
        x: np.ndarray
            A single sample for which to compute the perturbed probabilities.
        n_classes: int
            Number of classes in the data.
        perturbation_size: float
            The size of the perturbation to apply.
        perturbation_type: str
            The type of perturbation to apply.
        minimize: bool
            Whether to minimize or maximize the perturbed weights.

        Returns
        -------
        logprs: np.ndarray
            Log-probabilities of the class variable for the sample x.
    """
    if node.isleaf:
        logps = np.zeros(n_classes, dtype=np.float64)
        logps += node.logps[-1]

        for var in range(x.shape[0]):
            logps += node.logps[var][x[var]]

        return logps

    else:
        logps = np.zeros(n_classes, dtype=np.float64)

        split_feature = node.split.split_feature
        feat_values_left = node.split.threshold

        go_left = np.isin(np.array(x[split_feature]), feat_values_left)

        if go_left:
            logps += evaluate_gedt_perturbed(node.children[0], x, n_classes, perturbation_size, perturbation_type, minimize)
            weight = node.w[0]

        else:
            logps += evaluate_gedt_perturbed(node.children[1], x, n_classes, perturbation_size, perturbation_type, minimize)
            weight = node.w[1]

        pert_w = perturb_weights(weight, perturbation_size, perturbation_type, minimize)
        logps += np.log(pert_w)

        return logps


##############################
###### BUILD FUNCTIONS  ######
##############################

def build_tree(tree, parent, counts, ordered_ids, smoothing=1e-6):
    """
        Recursively splits the data, which contained in the tree object itself
        and is indexed by ordered_ids.

        Parameters
        ----------
        tree: Tree object
            The tree that is being built. Contains the data and parameters for building the tree.
        parent: TreeNode object
            The last node added to the tree, which will be the parent of the
            two nodes resulting from the split (if any) of this function call.
        counts: numpy array (int)
            The class counts of the samples reaching the parent node.
        ordered_ids: numpy array (int)
            The ids of the samples reaching the parent node.
        smoothing: float
            Additive smoothing (Laplace smoothing) for categorical data.

        Returns
        -------
        root: TreeNode object
            The root node of the tree.
        n_nodes: int
            The number of nodes in the tree.
    """
    root = TreeNode(0, counts, parent, ordered_ids, False)
    # root = SumNode(0, counts, parent, ordered_ids, False)
    queue = [root]
    n_nodes = 1
    np.random.seed(tree.random_state)

    data = np.concatenate([tree.X, np.expand_dims(tree.y, axis=1)], axis=1)

    while len(queue) > 0:
        node = queue.pop(0)
        split = find_best_split(node, tree, np.random.randint(1e6))

        if split is not None:
            node.split = split
            left_child = TreeNode(n_nodes, split.left_counts, node, split.left_ids, False)
            node.add_child(left_child)
            queue.append(left_child)
            n_nodes += 1

            right_child = TreeNode(n_nodes, split.right_counts, node, split.right_ids, False)
            node.add_child(right_child)
            queue.append(right_child)
            n_nodes += 1

        else:
            node.isleaf = True
            # n_points = len(ordered_ids)
            data_leaf = data[node.idx, :]
            scope = np.arange(data.shape[1], dtype=np.int64)
            # lp = np.sum(tree.n_feat_vals) * smoothing # LaPlace counts

            for var in scope:
                var_array = np.array([var])
                d = data_leaf[:, var_array].ravel()  # Project to scope
                d = np.asarray(d, np.int64)

                k = int(tree.ncat[var])

                if d.shape[0] > 0:
                    counts = bincount(d, k) + smoothing
                    # node.logcounts.append(np.log(counts))
                    p = counts/(d.shape[0] + k*smoothing)
                    node.ps.append(p)
                else:
                    counts = np.ones(k)
                    # node.logcounts.append(np.log(counts))
                    p = counts / k
                    node.ps.append(p)

                node.logps.append(np.log(np.asarray(node.ps[-1])))

        tree.depth = max(tree.depth, node.depth)
    return root, n_nodes


def build_gedt(X, y, bootstrap, n_feat_vals, n_classes, impurity_measure,
                 min_samples_split, min_samples_leaf, max_features, max_depth, random_state, smoothing=1e-6):
    """
        Fits a GeDT to (X, y).
        For the most part, the parameters match those of the Random Forest
        implementation in scikit-learn.
    """
    gedt = GeDT(n_feat_vals, n_classes, impurity_measure, min_samples_split, min_samples_leaf,max_features, max_depth, int(np.random.randint(1e6)))

    if bootstrap:
        Xtree_, ytree_, _ = resample_stratified(X, y, n_classes, gedt.random_state)
    else:
        Xtree_, ytree_ = X, y
    gedt.fit(Xtree_, ytree_, smoothing=smoothing)

    return gedt


##########################################


class TreeNode:
    """
        Class defining each node in a Decision Tree.

        Attributes
        ----------
        id: int
            Unique identifier for the node.
        counts: numpy array (int)
            The class counts of the samples reaching this node.
        n: int
            The total number of samples that reach this node.
        nchildren: int
            The number of children of this node.
        scope: numpy array
            The scope of the node, i.e., the variables that are used in the subtree rooted.
        sibling: TreeNode object
            A sibling of this node.
        w: numpy array
            The weights of the children of this node.
        logw: numpy array
            The log-weights of the children of this node.
        ps: list of numpy arrays
            The probabilities of the variables in the scope of this node.
        logps: list of numpy arrays
            The log-probabilities of the variables in the scope of this node.
        is_root: bool
            Whether this node is the root of the tree.
        parent: TreeNode object
            The parent of this node.
        idx: numpy array (int)
            The indices of the samples reaching this node in the original dataset.
        isleaf: bool
            Whether this node is a leaf node.
        split: Split object
            The split that was used to create the children of this node.
        left_child: TreeNode object
            The left child of this node.
        right_child: TreeNode object
            The right child of this node.

    """
    def __init__(self, id, counts, parent, idx, isleaf, scope=None, root_node=None):
        self.id = id
        self.counts = counts # counts of the target/class variable over all samples

        ## NEW ##
        self.n = np.sum(counts) #total number of samples that reach this node
        self.nchildren = 0
        self.scope = scope

        self.sibling = None

        # in case of a sum node (so not leaf node)
        self.w = None
        self.logw = None

        # leaf node
        self.ps = []
        self.logps = []

        if root_node is None:
            self.is_root = False
        else:
            self.is_root = root_node
        #########

        self.parent = parent
        self.idx = idx
        self.isleaf = isleaf
        self.split = None
        self.left_child = None
        self.right_child = None

        if parent is not None:
            self.depth = parent.depth + 1
        else:
            self.depth = 0

    @property
    def children(self):
        """ A list with all children. """
        children = []
        child = self.left_child
        while child is not None:
            children.append(child)
            child = child.sibling
        return children

    def add_sibling(self, sibling):
        self.sibling = sibling

    def add_child(self, child):
        # if parent has no children
        if self.left_child is None:
            # this node is it first child
            self.left_child = child
            self.right_child = child
        else:
            # the last (now right) child will have this node as sibling
            self.right_child.add_sibling(child)
            self.right_child = child

        self.nchildren += 1
        self.reweight()

    def reweight(self):
        children_n = np.array([c.n for c in self.children])
        n = np.sum(children_n)
        if n > 0:
            self.n = n
            self.w = np.divide(children_n.ravel(), self.n)
        else:
            self.n = 0
            self.w = np.ones(self.nchildren) * (1/self.nchildren)
        self.logw = np.log(self.w.ravel())


class GeDT:
    """
        Class that defines and evaluates a Generative Decision Tree (GeDT).

		Inspiration is taken from the repository https://github.com/AlCorreia/GeFs of Alvaro Correia, which implements the original GeF algorithm (Correia et al., 2020).

		Attributes
		----------
		X: numpy array of size n x m
            Numpy array comprising n realisations (instances) of m variables.
            These are the features used to learn the GeDT.
		y: numpy array of size n
			The class labels for each instance in X.
		scope: numpy array
			The scope of the GeDT, i.e., the variables that are used in the GeDT.
		n_feat_vals: numpy array
			The number of features values/categories/classes of each variable. One for continuous variables.
        n_classes: int
            The number of classes in the data.
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
        n_nodes: int
            The number of nodes in the GeDT.
        depth: int
            The depth of the tree.
        root_dt: TreeNode object
            The root node of the Decision Tree.
        root: Node object
            The root node of the GeDT.
		random_state: int
			The seed for the random number generator.


    """
    def __init__(self, n_feat_vals=None, n_classes=None, impurity_measure='gini', min_samples_split=2,
                 min_samples_leaf=1, max_features=None, max_depth=1e6, random_state=None):

        self.X = None
        self.y = None
        self.n_feat_vals = n_feat_vals
        self.n_classes = n_classes
        self.ncat = np.concatenate((self.n_feat_vals, np.asarray([self.n_classes], dtype=np.int64)))
        self.scope = None
        self.impurity_measure = impurity_measure
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_depth = max_depth
        self.n_nodes = 0
        self.depth = 0
        self.root = None

        self.root_dt = None
        if random_state is None:
            random_state = np.random.randint(1e6)  # Not ideal
        self.random_state = random_state


    def fit(self, X, y, smoothing=1e-6):
        """ Fits the Generative Decision Tree to (X, Y). """
        self.X = X
        self.y = y

        self.n_nodes = 0
        if self.max_features is None:
            self.max_features = X.shape[1]

        counts = np.bincount(y, minlength=self.n_classes)
        ordered_ids = np.arange(X.shape[0], dtype=np.int64)

        self.root, self.n_nodes = build_tree(self, None, counts, ordered_ids, smoothing=smoothing)

    def get_logprobs(self, X):
        """
            Returns the log-probabilities of the class variable for each instance in X.
        """
        root = self.root
        logps = evaluate_gedt(root, X, self.n_classes)
        return logps

    def get_perturbed_logprobs(self, x, perturbation_size, perturbation_type, minimize=True):
        """
            Returns the log-probabilities of the class variable for a single instance in X, after perturbing the weights in the GeDT.
        """
        root = self.root
        logps = evaluate_gedt_perturbed(root, x, self.n_classes, perturbation_size, perturbation_type, minimize)
        return logps

    def predict(self, X):
        """
            Returns the most likely class (highest number of counts) for
            each instance in X.
        """
        root = self.root
        logps = evaluate_gedt(root, X, self.n_classes)

        predictions = np.argmax(logps, axis=1)
        return predictions

    def get_node(self, id):
        """ Fetchs node by its id. """
        queue = [self.root_dt]
        while queue != []:
            node = queue.pop(0)
            if node.id == id:
                return node
            if node.split is not None:
                queue.extend([node.left_child, node.right_child])
        print("Node %d is not part of the network.", id)
