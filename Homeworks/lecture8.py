import numpy as np
import matplotlib.pyplot as plt
import math


def g(z):
    return 2*z-3


def layer(W, X, w0, f):
    return f(w0 + np.dot(W.transpose(), X))


X = np.array([[-1, -1],
              [1, -1],
              [-1, 1],
              [1, 1]]).T

Y = np.array([[1],
              [-1],
              [-1],
              [1]])


# # Caso 1
# w0 = np.zeros([2, 1])
# W = np.zeros([2, 2])
# print(np.append(layer(W, X, w0, g), Y).reshape(
#     X.shape[0] + 1, X.shape[1]), '\n')

# # Caso 2
# w0 = np.array([[1], [1]])
# W = np.array([[2, -2],
#               [2, -2]])
# print(np.append(layer(W, X, w0, g), Y).reshape(
#     X.shape[0] + 1, X.shape[1]), '\n')

# # Caso 3
# w0 = np.array([[1], [1]])
# W = np.array([[-2, 2],
#               [-2, 2]])
# print(np.append(layer(W, X, w0, g), Y).reshape(
#     X.shape[0] + 1, X.shape[1]), '\n')


# Caso 4
w0 = np.ones([2, 1])
W = np.array([[1, -1],
              [-1, 1]])


def g2(z):
    return 5*z - 2


F = layer(W, X, w0, g2)
print("f(z) = 5z - 2: \n", np.append(F,
                                     Y).reshape(X.shape[0] + 1, X.shape[1]), '\n')

# Plot
plt.scatter(F[np.tile(np.transpose(Y == 1), (F.shape[0], 1))].reshape((Y == 1).sum(), (Y == 1).sum())[0],
            F[np.tile(np.transpose(Y == 1), (F.shape[0], 1))].reshape(
                (Y == 1).sum(), (Y == 1).sum())[1],
            marker='+')
plt.scatter(F[np.tile(np.transpose(Y == -1), (F.shape[0], 1))].reshape((Y == -1).sum(), (Y == -1).sum())[0],
            F[np.tile(np.transpose(Y == -1), (F.shape[0], 1))
              ].reshape((Y == -1).sum(), (Y == -1).sum())[1],
            marker='_')
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('f(z) = 5z - 2')
plt.show()


def ReLU(z):
    return np.where(z < 0, 0, z)


F = layer(W, X, w0, ReLU)
print("f(z) = max{0, z}: \n", np.append(
    F, Y).reshape(X.shape[0] + 1, X.shape[1]), '\n')

# Plot
plt.scatter(F[np.tile(np.transpose(Y == 1), (F.shape[0], 1))].reshape((Y == 1).sum(), (Y == 1).sum())[0],
            F[np.tile(np.transpose(Y == 1), (F.shape[0], 1))].reshape(
                (Y == 1).sum(), (Y == 1).sum())[1],
            marker='+')
plt.scatter(F[np.tile(np.transpose(Y == -1), (F.shape[0], 1))].reshape((Y == -1).sum(), (Y == -1).sum())[0],
            F[np.tile(np.transpose(Y == -1), (F.shape[0], 1))
              ].reshape((Y == -1).sum(), (Y == -1).sum())[1],
            marker='_')
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('f(z) = max{0, z}')
plt.show()


def tanh(z):
    return np.tanh(z)


F = layer(W, X, w0, tanh)
print("f(z) = tanh(z): \n", np.append(
    F, Y).reshape(X.shape[0] + 1, X.shape[1]), '\n')

# Plot
plt.scatter(F[np.tile(np.transpose(Y == 1), (F.shape[0], 1))].reshape((Y == 1).sum(), (Y == 1).sum())[0],
            F[np.tile(np.transpose(Y == 1), (F.shape[0], 1))].reshape(
                (Y == 1).sum(), (Y == 1).sum())[1],
            marker='+')
plt.scatter(F[np.tile(np.transpose(Y == -1), (F.shape[0], 1))].reshape((Y == -1).sum(), (Y == -1).sum())[0],
            F[np.tile(np.transpose(Y == -1), (F.shape[0], 1))
              ].reshape((Y == -1).sum(), (Y == -1).sum())[1],
            marker='_')
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('f(z) = tanh(z)')
plt.show()


def lin(z):
    return z


F = layer(W, X, w0, lin)
print("f(z) = z: \n", np.append(F, Y).reshape(
    X.shape[0] + 1, X.shape[1]), '\n')

# Plot
plt.scatter(F[np.tile(np.transpose(Y == 1), (F.shape[0], 1))].reshape((Y == 1).sum(), (Y == 1).sum())[0],
            F[np.tile(np.transpose(Y == 1), (F.shape[0], 1))].reshape(
                (Y == 1).sum(), (Y == 1).sum())[1],
            marker='+')
plt.scatter(F[np.tile(np.transpose(Y == -1), (F.shape[0], 1))].reshape((Y == -1).sum(), (Y == -1).sum())[0],
            F[np.tile(np.transpose(Y == -1), (F.shape[0], 1))
              ].reshape((Y == -1).sum(), (Y == -1).sum())[1],
            marker='_')
plt.xlabel('f1')
plt.ylabel('f2')
plt.title('f(z) = z')
plt.show()
