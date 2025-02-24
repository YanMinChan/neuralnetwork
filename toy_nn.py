import numpy as np

# The activation function
def sig(x, dev=False):
    if (dev==True):
        return x*(1-x)
    return 1/(1 + np.exp(-x))

# The input and output array
# The NN
def nn(input, activation: Callable, seed=1):
    # Forward propagation
    np.random.seed(seed)
    w = np.random.rand(3,1)
    a = activation(np.dot(input, w))

    return input
