import numpy as np

### Functions for you to fill in ###


def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1

    Observations:
        There are cases where it's not a good idea to use the the closed-form solution:
            1. Matrix np.dot(np.dot(np.transpose(X), X) + lambda_factor*np.identity(X.shape[1])) is not invertible.
               np.dot(np.transpose(X),X) will not be invertible if it's not full rank or the determinant is zero,
               which man be checked using numpy.linalg.det() function. For cases where matrix is not invertible,
               a pseudoinverse is the next best option: numpy.linalg.pinv().
            2. Inverting the matrix is very slow with high dimensional features. np.dot(np.dot(np.transpose(X), X) + lambda_factor*np.identity(X.shape[1]))
               is an (d + 1) by (d + 1) matrix where (d + 1) is the number of feature. Time complexity for inverting the matrix is O((d + 1)^3).
    """
    # YOUR CODE HERE

    # Check if np.dot(np.dot(np.transpose(X), X) + lambda_factor*np.identity(X.shape[1])) is invertible
    if np.linalg.det(np.dot(np.transpose(X), X) + lambda_factor*np.identity(X.shape[1])) != 0:
        return np.dot(np.linalg.inv(np.dot(np.transpose(X), X) + lambda_factor*np.identity(X.shape[1])), np.dot(np.transpose(X), Y))
    return np.nan


def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
