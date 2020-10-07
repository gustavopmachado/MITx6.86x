import numpy as np

# Linear Regressions Exercises


def empirical_risk(theta, X, Y, loss):
    loss_sum = 0
    for i, feature_vector in enumerate(X):
        loss_sum += loss(Y[i] - np.dot(np.transpose(theta), feature_vector))
    return loss_sum/X.shape[0]


def hinge_loss(z):
    if z < 1:
        return 1 - z
    return 0


def lms_loss(z):
    return z**2/2


if __name__ == "__main__":
    # Defines the Feature Matrix and its Labels
    X = np.array([[1, 0, 1],
                  [1, 1, 1],
                  [1, 1, -1],
                  [-1, 1, 1]])

    Y = np.array([2,
                  2.7,
                  -0.7,
                  2])

    # Define the Parameters
    theta = np.array([0, 1, 2])

    # Calculate the Risk
    R_hinge = empirical_risk(theta, X, Y, hinge_loss)
    R_lms = empirical_risk(theta, X, Y, lms_loss)

    print('Hinge: {}'.format(R_hinge))
    print('LMS: {}'.format(R_lms))
