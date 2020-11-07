import numpy as np

# It's possible, given the mistakes of each training data, to calculate the parameters when using Linear Perceptron Algorithm as follows:
#
# \theta := theta_{initial} + \sum_{i=1}^{n} \beta^{i}y^{i}x^{i}
# \theta_{0} := theta_{0, initial} + \sum_{i=1}^{n} \beta^{i}y^{i}
#
# where \beta^{i} is the number of mistakes of ith training point

# Initialize the parameters as 0
theta_exam, theta_0_exam = np.zeros((1, 2)), 0

# Number os mistakes
history_exam = {'x1': 1, 'x2': 9, 'x3': 10, 'x4': 5,
                'x5': 9, 'x6': 11, 'x7': 0, 'x8': 3, 'x9': 1, 'x10': 1}

# Training set
x = np.array([[0, 0],
              [2, 0],
              [3, 0],
              [0, 2],
              [2, 2],
              [5, 1],
              [5, 2],
              [2, 4],
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

# Evaluate the parameters
for i in range(len(history_exam)):
    theta_exam += history_exam.get('x' + str(i + 1))*y[i]*x[i]
    theta_0_exam += history_exam.get('x' + str(i + 1))*y[i]

print(f"Theta Exam: {theta_exam} | Theta 0 Exam: {theta_0_exam} \n")
