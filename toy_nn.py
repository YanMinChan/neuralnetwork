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
y = np.array([[0, 1, 1, 0]]).T

# The NN
def nn(input, output, activation, seed=1):
    # Forward propagation
    np.random.seed(seed)
    # Assign random weight with mean 0
    w1 = 2*np.random.rand(3,4) - 1
    w2 = 2*np.random.rand(4,1) - 1
    for iter in range(60000):
        l0 = input
        l1 = activation(np.dot(l0, w1))
        l2 = activation(np.dot(l1, w2))


        l2_error = output - l2
        l2_delta = l2_error * activation(l2, True)

        l1_error = l2_delta.dot(w2.T)
        l1_delta = l1_error * activation(l1, True)

        w1 += np.dot(l0.T, l1_delta)
        w2 += np.dot(l1.T, l2_delta)

    return l2

print(nn(X, y, sig))
