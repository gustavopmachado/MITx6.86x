import numpy as np


def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """

    A = np.random.random_sample([n, 1])
    return A


def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """

    A = np.random.random_sample([h, w])
    B = np.random.random_sample([h, w])
    s = A + B

    return A, B, s


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """

    s = np.linalg.norm(A+B)

    return s


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """

    # Dot product between the inputs and weights vectors
    r = np.sum(np.dot(np.transpose(weights), inputs))

    # Apply a non-linear function to the result
    z = np.array([np.tanh(np.array([r]))])

    return z


def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """

    if x <= y:
        return x * y
    return x / y


def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y.
    """

    # Vectorize the scalar function
    f = np.vectorize(scalar_function)

    return f(x, y)


def main():

    x = np.random.random_sample([2, 1])
    y = np.random.random_sample([2, 1])
    z = vector_function(x, y)
    print(z)


if __name__ == "__main__":
    main()
