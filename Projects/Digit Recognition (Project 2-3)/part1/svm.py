import numpy as np
from sklearn.svm import LinearSVC


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x, C):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point

    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
    """
    # Initialize the 'One-vs-Rest' LinearSVC class considering the hinge loss formulation
    svc = LinearSVC(C=C, random_state=0)

    # Trains the LinearSVC Class
    svc.fit(train_x, train_y)

    # Prediction
    return svc.predict(test_x)


def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    
    Reference: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
    
    Obs.: The LinearSVC is robust enough to use the multi-class SVM when the input labels are multi-class
    """
    # Initialize the 'One-vs-Rest' LinearSVC class considering the hinge loss formulation
    svc = LinearSVC(C=0.1, random_state=0)

    # Trains the LinearSVC Class
    svc.fit(train_x, train_y)

    # Prediction
    return svc.predict(test_x)


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)
