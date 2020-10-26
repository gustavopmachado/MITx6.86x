import numpy as np
import matplotlib.pyplot as plt

##########################################################################
#  1. Neural Networks
##########################################################################

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
