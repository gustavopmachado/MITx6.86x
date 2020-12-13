import numpy as np


def states():
    return np.arange(5)


def actions():
    return {'left', 'stay', 'right'}


def rewards(S, A):
    # a \in A = {left, stay, right}
    R = np.zeros((S.shape[0], len(A), S.shape[0]))

    # Reward staying in the rightmost state before halting
    R[-1, 1, -1] = 1

    return R


# def transitions():
#     T = np.zeros((R.shape[0], 3, R.shape[0]))  # a \in A = {left, stay, right}

#     # "If the agent chooses to stay at the location,
#     #  such an action is successful with probability 1/2"
#     T[:, 1, :] = 0.5 * np.identity(5)

#     # "If the agent is at the leftmost or rightmost grid
#     # location it ends up at its neighboring grid
#     # location with probability 1/2"
#     T[0, 2, 1] = 0.5
#     T[-1, 0, -2] = 0.5

#     # "If the agent is at any of the inner grid locations
#     # it has a probability 1/4 each of ending up at either
#     # of the neighboring locations"
#     T[1, 0, 0] = T[1, 2, 2] = 1/4
#     T[2, 0, 1] = T[2, 2, 3] = 1/4
#     T[3, 0, 2] = T[3, 2, 4] = 1/4

#     # If the agent chooses to move (either left or right) at
#     # any of the inner grid locations, such an action is successful
#     # with probability  1/3  and with probability  2/3  it fails to move
#     T[1, 1, 1] = T[2, 1, 2] = T[3, 1, 3] = T[1, 1, 1] + T[1, 0, 0] * 2/3

#     T[1, 0, 0], T[1, 2, 2] = T[1, 0, 0] * 1/3, T[1, 2, 2] * 1/3
#     T[2, 0, 1], T[2, 2, 3] = T[2, 0, 1] * 1/3, T[2, 2, 3] * 1/3
#     T[3, 0, 2], T[3, 2, 4] = T[3, 0, 2] * 1/3, T[3, 2, 4] * 1/3

#     return T


def transitions(R, A):
    # a \in A = {left, stay, right}
    # T(s, a, s')
    T = np.zeros((R.shape[0], len(A), R.shape[0]))

    # Right Move with 100% probability
    for i in range(R.shape[0] - 1):
        T[i, -1, i+1] = 1

    # Stay in place in the last move
    T[-1, 1, -1] = 1

    return T


def value_iteration_update(T, R, gamma, V):
    return np.amax(np.sum(T * (R + gamma * V), axis=2), axis=1)


if __name__ == "__main__":

    # Parameter Initialization
    gamma = 0.5
    S = states()
    A = actions()
    R = rewards(S, A)
    T = transitions(R, A)

    # Run the Value Iteration
    V = np.zeros(5)

    for k in range(5):
        V[:5 - k] = value_iteration_update(T, R, gamma, V)[:5 - k]

    print(f"V: {V}")
