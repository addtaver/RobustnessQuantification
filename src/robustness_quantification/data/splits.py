import numpy as np


def split_train_test(X, y, X_test=None, y_test=None, fraction_test=0.4, max_size_test_set=500):
    """Return (X_train, y_train, X_test, y_test).

    If the dataset has no predefined test set, the data is randomly permuted (using the global
    NumPy RNG) and the first ``min(fraction_test * n, max_size_test_set)`` points become the test set.
    Otherwise the provided test set is truncated to ``max_size_test_set`` points.
    """
    if X_test is None:
        test_set_size = min(int(fraction_test * X.shape[0]), max_size_test_set)

        perm = np.random.permutation(len(y))
        X, y = X[perm], y[perm]

        return X[test_set_size:], y[test_set_size:], X[:test_set_size], y[:test_set_size]

    return X, y, X_test[:max_size_test_set], y_test[:max_size_test_set]


def introduce_label_noise(y_train, number_of_classes, noise_fraction=0.05, random_state=42):
    """Flip a fraction of the labels to a different, uniformly chosen class."""
    np.random.seed(random_state)
    n_samples = len(y_train)
    n_noisy = int(noise_fraction * n_samples)
    noisy_indices = np.random.choice(n_samples, n_noisy, replace=False)

    y_train_noisy = y_train.copy()
    for index in noisy_indices:
        current_label = y_train[index]
        possible_labels = np.arange(number_of_classes)
        possible_labels = possible_labels[possible_labels != current_label]
        y_train_noisy[index] = np.random.choice(possible_labels)

    return y_train_noisy
