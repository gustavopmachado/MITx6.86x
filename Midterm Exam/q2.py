import numpy as np

# It's possible, given the mistakes of each training data, to calculate the parameters when using Kernel Perceptron Algorithm as follows:
#
# [Writen in LaTeX]
# \theta := theta_{initial} + \sum_{i=1}^{n} \beta^{i}y^{i}\phi(x^{i})
# \theta_{0} := theta_{0, initial} + \sum_{i=1}^{n} \beta^{i}y^{i}
#
# where \beta^{i} is the number of mistakes of ith training point

# Initialize the parameters as 0
theta_exam, theta_0_exam = np.zeros(3), 0

# Number os mistakes
history_exam = {'x1': 1, 'x2': 65, 'x3': 11, 'x4': 31,
                'x5': 72, 'x6': 30, 'x7': 0, 'x8': 21, 'x9': 4, 'x10': 15}

# Training set
x = np.array([[0, 0],
              [2, 0],
              [1, 1],
              [0, 2],
              [3, 3],
              [4, 1],
              [5, 2],
              [1, 4],
              [4, 4],
              [5, 5]])

y = np.array([[-1],
              [-1],
              [-1],
              [-1],
              [-1],
              [1],
              [1],
              [1],
              [1],
              [1]])


def Kernel(x):
    return np.array([x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2])


# Evaluate the parameters
for i in range(len(history_exam)):
    theta_exam += history_exam.get('x' + str(i + 1))*y[i]*Kernel(x[i])
    theta_0_exam += history_exam.get('x' + str(i + 1))*y[i]

print(
    f"Kernel Vector:\n {np.array([Kernel(x[i]) for i in range(len(history_exam))])}\n")
print(f"Theta Exam: {theta_exam} | Theta 0 Exam: {theta_0_exam} \n")


# Classify the Training Points considering the Parameters above:
prediction = np.empty((y.shape))
for i, point in enumerate(x):
    phi = Kernel(x[i])
    prediction[i] = np.where(np.dot(theta_exam, phi) + theta_0_exam < 0, -1, 1)

if (y == prediction).sum() == y.shape[0]:
    print(
        f"The Kernel Perceptron correctly classified all points")
else:
    print(
        f"The Kernel Perceptron incorrectly classified {(y != prediction).sum()} points")
