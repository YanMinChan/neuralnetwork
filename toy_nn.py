import numpy as np
from collections.abc import Callable

# The activation function
def sig(x, dev=False):
    if (dev==True):
        return x*(1-x)
    return 1/(1 + np.exp(-x))

# The input and output array
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])
y = np.array([[0, 0, 1, 1]]).T

# The NN
def nn(input, output, activation, seed=1):
    # Forward propagation
    np.random.seed(seed)
    w = 2*np.random.rand(3,1) - 1
    for iter in range(10000):
        l0 = input
        l1 = activation(np.dot(input, w))

        l1_error = output - l1

        l1_delta = l1_error * activation(l1, True)

        w += np.dot(l0.T, l1_delta)

    return l1

print(nn(X, y, sig))
