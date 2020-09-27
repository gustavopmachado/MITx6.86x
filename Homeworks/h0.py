import numpy as np


def parameter_initialization(x):
    """
    Initiliaze the Perceptron parameters theta and theta_not as zero vector and scalar respectively

    Args:
        x - Training set (numpy array: n x d)

    Returns (in this order):
        theta - Parameter vector (numpy array: n x 1)
        theta_not - Offset scalar (numpy array: 1 x 1)
    """

    assert isinstance(
        x, np.ndarray), "Training set, x, should be a numpy array"

    return np.zeros(x.shape[1]), np.array([0])


def perceptron_step(theta, theta_not, x, y):
    """
    Performs the step procedure of Perceptron algorithm updating the parameters if condition is verified

    Args
        theta - Parameter vector (numpy array: n x 1)
        theta_not - Offset scalar (numpy array: 1 x 1)
        x - One training vector (numpy array: n x 1)
        y - x's label (numpy array: 1 x 1)

    Returns (in this order):
        theta - Parameter vector (numpy array: n x 1)
        theta_not - Offset scalar (int)
    """

    assert all([isinstance(elm, np.ndarray) for elm in [theta, theta_not, x, y]]
               ), "Arguments should be of type numpy array"

    if y*(np.dot(np.transpose(theta), x) + theta_not) <= 0:
        theta = theta + y*x
        theta_not = theta_not + y

        return theta, theta_not

    return theta, theta_not


def perceptron(x, y, theta=None, theta_not=None, **kwargs):
    """
    Perceptron algorithm runs until full convergence, which means when there's no update in the run through

    Args:
        x - Training set (numpy array: n x d)
        y - Training set label (numpy array: n x 1)

    Optional:
        order - Ordering list that the algorithm will perform through (list, range or numpy array: n x 1)

    Returns (in this order):
        theta - Parameter vector (numpy array: n x 1)
        theta_not - Offset scalar (numpy array: 1 x 1)
    """

    assert all([isinstance(x, np.ndarray) for elm in [x, y]]
               ), "Training set (x) and its labels (y) should be a numpy array"

    # Defines the algorithm ordering and convergence status
    order = kwargs.get('order', range(x.shape[0]))

    assert isinstance(order, range) or isinstance(order, list) or isinstance(
        order, np.ndarray), "order should be of either type list or numpy array"

    # Initialize the Perceptron parameters
    if theta is None and theta_not is not None:
        assert isinstance(theta, np.ndarray), "theta must be of type array"
        assert theta.shape[0] == x.shape[0], "theta size differs from training set size"
        theta, _ = parameter_initialization(x)

    elif theta is None and theta_not is not None:
        assert isinstance(theta_not, np.ndarray), "theta must be of type array"
        assert theta.shape[0] == 1, "theta_not must be a 1 x 1 numpy array"
        _, theta_not = parameter_initialization(x)

    else:
        theta, theta_not = parameter_initialization(x)

    # Initiliaze the list to store the parameters progress
    theta_prog = np.array(theta)

    # Initialize the dictionary to store each point update history
    update_history = dict(
        zip([f"x{i + 1}" for i in range(x.shape[0])],
            list(np.zeros(x.shape[0]))))

    # Runs the Perceptron Algorithms
    while True:
        miss_classified = 0
        for i in order:
            theta, theta_not = perceptron_step(theta, theta_not, x[i], y[i])

            if any(theta != theta_prog[-1]):
                # Append the new theta in order to track its progress
                theta_prog = np.vstack((theta_prog, theta))

                # Updates the training point history
                update_history[f"x{i + 1}"] = update_history.get(
                    f"x{i + 1}") + 1

                # Updates the flag as miss classified
                miss_classified = 1

        # Print the current update history and the related theta
        print(f"Theta: {theta} | Theta 0: {theta_not}")
        print(f"History: {update_history} \n")

        # Homework Part 2 Stop Criteria
        # if update_history["x1"] == 1 and update_history["x2"] == 0 and update_history["x3"] == 2 and update_history["x4"] == 1 and update_history["x5"] == 0:
        #   break

        # Check if no change in parameters was made in the run through
        if miss_classified == 0:
            break

    return theta, theta_not, theta_prog


if __name__ == "__main__":

    # # 1) Without Offset parameter

    # # Defines the Training Set
    # x = np.array([[-1, -1],
    #               [1, 0],
    #               [-1, 10]])
    # y = np.array([[1],
    #               [-1],
    #               [1]])

    # # Defines the ordering in which the Perceptron will go through the training set
    # order = np.array([1, 2, 3]) - np.ones(x.shape[0]
    #                                       ).astype(int)  # Starting at x(1)
    # # order = np.array([2, 3, 1]) - np.ones(x.shape[0]
    # #                                       ).astype(int)  # Starting at x(2)

    # # Runs the Perceptron
    # theta, theta_not, theta_prog = perceptron(x, y, order=order)

    # print(f"Training Set: {x} \n")
    # print(f"Theta Progress: {theta_prog} \n")
    # print(f"Theta: {theta} | Theta 0: {theta_not} \n")

    # # 2) With offset parameter
    # # Defines the Training Set
    # x = np.array([[-4, 2],
    #               [-2, 1],
    #               [-1, -1],
    #               [2, 2],
    #               [1, -2]])
    # y = np.array([[1],
    #               [1],
    #               [-1],
    #               [-1],
    #               [-1]])

    # # Runs the Perceptron
    # theta, theta_not, theta_prog = perceptron(x, y)

    # print(f"Training Set: {x} \n")
    # print(f"Theta Progress: {theta_prog} \n")
    # print(f"Theta: {theta} | Theta 0: {theta_not} \n")

    # 3) Perceptron for the following Training Set
    # Defines the Training Set
    x = np.array([[-1, 1],
                  [1, -1],
                  [1, 1],
                  [2, 2]])
    y = np.array([[1],
                  [1],
                  [-1],
                  [-1]])

    # Runs the Perceptron
    theta, theta_not, theta_prog = perceptron(x, y)

    print(f"Training Set: {x} \n")
    print(f"Theta Progress: {theta_prog} \n")
    print(f"Theta: {theta} | Theta 0: {theta_not} \n")
