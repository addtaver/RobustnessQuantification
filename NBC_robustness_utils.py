import numpy as np


#-------
# Helper: monotone predicate for bisection
#
# For fixed sample and class c:
#   score_c(eps) = log(prior_c + t) + sum_j log(like_cj(x_j) + t)
# where t = eps / (1 - eps)
#
# Since t increases with eps and log(a+t) increases with t, score_c is monotone
# increasing in eps. Therefore:
#   exists competitor with score >= winner_score
# is a monotone predicate → bisection applies.
# -------
def robustness_check_eps(eps, params_mat, win_idx, target):
    """
    Returns True if any class != win_idx reaches/exceeds `target` at epsilon=eps.

    params_mat: shape (K, n_features+1)
        Each row contains [prior_c, like_c0(x0), ..., like_c,d-1(xd-1)] for class c
    """
    t = eps / (1.0 - eps)                 # monotone transform, eps in [0,1)
    scores = np.log(params_mat + t).sum(axis=1)  # vectorized over classes
    scores[win_idx] = -np.inf             # exclude the winning class
    return np.any(scores >= target)


def robustness_check_TV(delta, prior, likel_vals, win_idx):
    """
    Returns True if P(c)*lambda - P(c^)*mu + delta*(lambda - mu) > 0 for at least one non-winning class,
    where lambda is the upper probability of the likel_vals for the non-winning class and mu is the lower probability of the likel_vals for the winning class.

    params_mat: shape (K, n_features+1)
        Each row contains [prior_c, like_c0(x0), ..., like_c,d-1(xd-1)] for class c
    """
    # calculate mu: the lower probabilities of the winning class
    mu = np.maximum(likel_vals[win_idx] - delta, 0.0)  # subtract delta from winning class, but floor at 0
    mu_prod = np.prod(mu) # take product of all values in mu, this gives the lower probability of the features for the winning class

    # take product with the prior for the winning class
    win_prior_mu = prior[win_idx] * mu_prod

    # add delta to all values in likel_vals, but cap at 1
    lambd = np.minimum(likel_vals + delta, 1.0) # add delta to all values, but cap at 1
    # take product of all values in lambd for each class, this gives the upper probability of the features for each class
    lambd_prod = np.prod(lambd, axis=1)

    # calculate the product with the prior for all classes
    prior_lambda = prior[:, None] * lambd_prod[:, None]

    # for each class except the winning class, check if the following condition holds:
    # prior_lambda - win_prior_mu + delta * (lambda_prod - mu_prod) > 0
    for c in range(len(prior)):
        if c == win_idx:
            continue

        if prior_lambda[c] - win_prior_mu + delta * (lambd_prod[c] - mu_prod) > 0:
            return True
    return False


def robustness_check_chi2(delta, prior, likel_vals, win_idx):
    """
    Returns True if E_{lower}(Ic^*alplha - Ic*beta) < 0 for at least one non-winning class,
    where lambda is the upper probability of the likel_vals for the non-winning class and mu is the lower probability of the likel_vals for the winning class.

    params_mat: shape (K, n_features+1)
        Each row contains [prior_c, like_c0(x0), ..., like_c,d-1(xd-1)] for class c
    """
    # calculate mu: the lower probabilities of the winning class
    # for each value in likel_vals for the winning class, subtract sqrt(delta * prob*(1-prob)), but floor at 0
    alpha = np.maximum(likel_vals[win_idx] - np.sqrt(delta * likel_vals[win_idx] * (1 - likel_vals[win_idx])), 0.0)
    alpha_prod = np.prod(alpha) # take product of all values in mu, this gives the lower probability of the features for the winning class

    # take product with the prior for the winning class
    win_prior_alpha = prior[win_idx] * alpha_prod

    # for all values in likel_vals, add sqrt(delta * prob*(1-prob)), but cap at 1
    beta = np.minimum(likel_vals + np.sqrt(delta * likel_vals * (1 - likel_vals)), 1.0)
    # take product of all values in beta for each class, this gives the upper probability of the features for each class
    beta_prod = np.prod(beta, axis=1)

    # calculate the product with the prior for all classes
    prior_beta = prior[:, None] * beta_prod[:, None]

    # for each class except the winning class, check if the following condition holds:
    # prior_beta - win_prior_mu + sqrt(delta * var) > 0
    for c in range(len(prior)):
        if c == win_idx:
            continue

        var = beta_prod[c] * prior_beta[c] * (1 - prior[c]) + alpha_prod * win_prior_alpha * (1 - prior[win_idx]) + 2 * prior_beta[c] * win_prior_alpha

        if prior_beta[c] - win_prior_alpha + np.sqrt(delta * var) > 0:
            return True
    return False