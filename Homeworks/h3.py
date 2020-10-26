import numpy as np
import matplotlib.pyplot as plt

##########################################################################
#  1. Neural Networks
##########################################################################


def NN():

    # Input Layer
    X = np.array([[3, 14]])

    # Input - Hidden Layer Weights
    W = np.array([[1, 0, -1],
                  [0, 1, -1],
                  [-1, 0, -1],
                  [0, -1, -1]])

    # Hidden Layer - Output Weights
    V = np.array([[1, 1, 1, 1, 0],
                  [-1, -1, -1, -1, 2]])

    # Hidden Layer Activation
    Z = np.inner(X, W.transpose()[:-1].T) + W.transpose()[-1].T
    print("Hidden Layer Z's: ", Z, '\n')

    # Output Layer Activation
    U = np.dot(np.maximum(Z, 0), V.transpose()[:-1].T.T) + V.transpose()[-1].T
    print("Output Layer U's: ", U, '\n')

    # Softmax Output Layer Activation
    print("ReLU Output Layer U's: ", U, '\n')
    O = np.exp(np.maximum(U, 0))/np.exp(np.maximum(U, 0)).sum()
    print("Softmax Output Layer O's: ", O, '\n')

    # Plot the Hidden Layer Decision Boundary
    # Those Boundary are defined where the Z's on the Hidden Layer are 0
    for i in range(W.shape[0]):
        if W.transpose()[0].transpose()[i] != 0:
            xx = np.linspace(-2.5, 2.5)
            yy = (-W.transpose()[-1].transpose()[i] - W.transpose()
                  [1].transpose()[i]*xx)/W.transpose()[0].transpose()[i]
            plt.plot(xx, yy)
        else:
            plt.vlines((-W.transpose()[-1].transpose()[i] /
                        W.transpose()[1].transpose()[i]), -2.5, 2.5)
    plt.show()

    return None


##########################################################################
#  2. Recurrent Neural Networks (Long-short Term Memory RNN)
##########################################################################


def sigmoid(x):
    """
    """
    if x >= 1:
        return 1
    elif x <= -1:
        return 0
    return 1/(1 + np.exp(-x))


def RNN_step(Wfh, Wfx, Wih, Wix, Woh, Wox, Wch, Wcx, bf, bi, bo, bc, h, c, x):
    """
    """

    f = sigmoid(Wfh*h + Wfx*x + bf)
    i = sigmoid(Wih*h + Wix*x + bi)
    o = sigmoid(Woh*h + Wox*x + bo)
    c = f*c + i*np.tanh(Wch*h + Wcx*x + bc)
    h = o*np.tanh(c)

    return c, h


def RNN(X, h0, c0, T):
    # RNN Parameters
    Wfh, Wih, Woh, Wfx, bo, bc = 0, 0, 0, 0, 0, 0
    Wix, Wox, bi = 100, 100, 100
    Wch, bf = -100, -100
    Wcx = 50

    # Initialize the hidden state array
    C = np.append(c0, np.zeros([1, T]))
    H = np.append(h0, np.zeros([1, T]))

    for t in range(T):
        C[t + 1], H[t + 1] = RNN_step(Wfh, Wfx, Wih, Wix, Woh, Wox,
                                      Wch, Wcx, bf, bi, bo, bc, H[t], C[t], X[t])
        C[t + 1], H[t + 1] = np.round(C[t + 1]), np.round(H[t + 1])

    return C[1:], H[1:]


if __name__ == "__main__":

    # X = np.array([0, 0, 1, 1, 1, 0])
    X = np.array([1, 1, 0, 1, 1])
    h0, c0 = 0, 0
    T = X.shape[0]
    C, H = RNN(X, h0, c0, T)
    print("C: ", C)
    print("H: ", H)
