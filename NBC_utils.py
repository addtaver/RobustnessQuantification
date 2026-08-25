import numpy as np

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