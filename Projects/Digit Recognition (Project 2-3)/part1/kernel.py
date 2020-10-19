import numpy as np

### Functions for you to fill in ###


def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    return (np.dot(X, Y.transpose()) + c)**p


def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # TODO: Work on the pairwise distance
    # dist = (X.reshape((X.shape[0], X.shape[1], 1)) - Y.transpose()).sum(axis=1)

    # L2 Vectorized distance between matrices
    dist = -2 * np.dot(X, Y.transpose()) + np.sum(Y**2,
                                                  axis=1) + np.sum(X**2, axis=1)[:, np.newaxis]
    return np.exp(-gamma*dist)
